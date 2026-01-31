"""
RAG Pipeline for B-PLIS-RAG.

Implements the complete Retrieval-Augmented Generation pipeline
with support for ReFT interventions and activation steering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.config import get_config
from src.model_loader import load_model, generate_text
from src.retriever import FAISSRetriever
from src.reft import ReFTIntervention, ReFTHook, create_reft_intervention
from src.activation_steering import ActivationSteering
from src.data_handler import Document

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    query: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": self.sources,
            "scores": self.scores,
            "metadata": self.metadata,
        }


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    model_name: str = "t5-base"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    max_context_length: int = 1024
    max_new_tokens: int = 100
    use_reft: bool = False
    use_steering: bool = False
    reft_layer: int = 6
    reft_dim: int = 16
    steering_layer: int = 6
    steering_multiplier: float = 2.0
    
    @classmethod
    def from_global_config(cls) -> "RAGConfig":
        """Create from global configuration."""
        config = get_config()
        return cls(
            model_name=config.model.name,
            embedding_model=config.retriever.embedding_model,
            top_k=config.retriever.top_k,
            max_new_tokens=config.model.max_new_tokens,
            use_reft=True,
            use_steering=True,
            reft_layer=config.reft.target_layer,
            reft_dim=config.reft.intervention_dim,
            steering_layer=config.steering.steering_layer,
            steering_multiplier=config.steering.multiplier,
        )


class RAGPipeline:
    """
    Complete RAG pipeline with ReFT and activation steering.
    
    Combines:
    - Document retrieval (FAISS + sentence embeddings)
    - T5 generation with context
    - Optional ReFT intervention for faithfulness
    - Optional activation steering for context focus
    
    Example:
        >>> pipeline = RAGPipeline(model_name="t5-base", use_reft=True)
        >>> pipeline.index_documents(documents)
        >>> response = pipeline.query("What is a breach of contract?")
        >>> print(response.answer)
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_reft: bool = False,
        use_steering: bool = False,
        reft_layer: int = 6,
        reft_dim: int = 16,
        steering_layer: int = 6,
        steering_multiplier: float = 2.0,
        device: Optional[str] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Name of the T5 model.
            embedding_model: Sentence transformer for embeddings.
            use_reft: Whether to use ReFT interventions.
            use_steering: Whether to use activation steering.
            reft_layer: Layer for ReFT intervention.
            reft_dim: Dimension of ReFT intervention.
            steering_layer: Layer for activation steering.
            steering_multiplier: Steering strength.
            device: Device to use.
            config: Optional RAGConfig to use instead of parameters.
        """
        if config:
            model_name = config.model_name
            embedding_model = config.embedding_model
            use_reft = config.use_reft
            use_steering = config.use_steering
            reft_layer = config.reft_layer
            reft_dim = config.reft_dim
            steering_layer = config.steering_layer
            steering_multiplier = config.steering_multiplier
        
        self.model_name = model_name
        self.use_reft = use_reft
        self.use_steering = use_steering
        self.steering_multiplier = steering_multiplier
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = load_model(model_name, device=device)
        self.device = next(self.model.parameters()).device
        
        # Initialize retriever
        logger.info(f"Initializing retriever with {embedding_model}")
        self.retriever = FAISSRetriever(
            embedding_model=embedding_model,
            device=str(self.device) if torch.cuda.is_available() else "cpu",
        )
        
        # Initialize ReFT if enabled
        self.reft_intervention: Optional[ReFTIntervention] = None
        self.reft_hook: Optional[ReFTHook] = None
        
        if use_reft:
            logger.info(f"Initializing ReFT intervention (layer={reft_layer}, dim={reft_dim})")
            self.reft_intervention, self.reft_hook = create_reft_intervention(
                self.model,
                intervention_dim=reft_dim,
                target_layer=reft_layer,
            )
        
        # Initialize steering if enabled
        self.steerer: Optional[ActivationSteering] = None
        
        if use_steering:
            logger.info(f"Initializing activation steering (layer={steering_layer})")
            self.steerer = ActivationSteering(
                self.model,
                self.tokenizer,
                layer=steering_layer,
                device=self.device,
            )
        
        # Configuration
        self.top_k = config.top_k if config else 5
        self.max_context_length = config.max_context_length if config else 1024
        self.max_new_tokens = config.max_new_tokens if config else 100
        
        logger.info("RAG pipeline initialized")
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index.
            batch_size: Batch size for encoding.
        """
        logger.info(f"Indexing {len(documents)} documents")
        self.retriever.index_documents(documents, batch_size=batch_size)
    
    def compute_steering_vector(
        self,
        examples: List[Dict[str, str]],
    ) -> None:
        """
        Compute activation steering vector from examples.
        
        Args:
            examples: List of dicts with 'query' and 'context' keys.
        """
        if self.steerer is None:
            raise RuntimeError("Steering not enabled. Initialize with use_steering=True")
        
        self.steerer.compute_from_examples(examples)
        logger.info("Steering vector computed")
    
    def _build_prompt(
        self,
        query: str,
        contexts: List[str],
        template: str = "default",
    ) -> str:
        """
        Build the prompt for generation.
        
        Args:
            query: User query.
            contexts: Retrieved context snippets.
            template: Prompt template to use.
            
        Returns:
            Formatted prompt string.
        """
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Truncate if too long
        if len(combined_context) > self.max_context_length:
            combined_context = combined_context[:self.max_context_length] + "..."
        
        if template == "default":
            # Better prompt for T5/Flan-T5
            prompt = f"Answer the following question based on the context.\n\nContext:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"
        elif template == "legal":
            prompt = f"Based on the following legal document excerpts, provide a precise answer.\n\nDocuments:\n{combined_context}\n\nLegal Question: {query}\n\nAnswer:"
        elif template == "simple":
            prompt = f"{combined_context}\n\nQ: {query}\nA:"
        else:
            prompt = f"Use context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
        
        return prompt
    
    @torch.no_grad()
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        template: str = "default",
        return_sources: bool = True,
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question.
            top_k: Number of documents to retrieve.
            template: Prompt template.
            return_sources: Whether to include source documents.
            
        Returns:
            RAGResponse with answer and metadata.
        """
        top_k = top_k or self.top_k
        
        # Retrieve relevant documents with longer snippets
        retrieved = self.retriever.retrieve_with_snippets(question, top_k=top_k, snippet_length=1000)
        
        # Extract contexts
        contexts = [r["snippet"] for r in retrieved]
        scores = [r["score"] for r in retrieved]
        
        # Build prompt
        prompt = self._build_prompt(question, contexts, template)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # Increased from 512 to allow more context
        ).to(self.device)
        
        # Generate with optional steering
        try:
            if self.use_reft and self.reft_hook:
                self.reft_hook.register()
            
            if self.use_steering and self.steerer and self.steerer.steering_vector is not None:
                self.steerer.apply_manual(self.steering_multiplier)
            
            # Generate with greedy decoding for extractive QA
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Longer to capture full definitions
                do_sample=False,
                early_stopping=True,
            )
        
        finally:
            # Cleanup hooks
            if self.use_reft and self.reft_hook:
                self.reft_hook.remove()
            
            if self.use_steering and self.steerer:
                self.steerer.remove_manual()
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Build response
        sources = []
        if return_sources:
            for r in retrieved:
                sources.append({
                    "id": r["id"],
                    "source": r["source"],
                    "snippet": r["snippet"][:200] + "..." if len(r["snippet"]) > 200 else r["snippet"],
                    "score": r["score"],
                })
        
        return RAGResponse(
            answer=answer,
            query=question,
            sources=sources,
            scores=scores,
            metadata={
                "model": self.model_name,
                "top_k": top_k,
                "use_reft": self.use_reft,
                "use_steering": self.use_steering,
                "prompt_length": len(prompt),
            },
        )
    
    def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        template: str = "default",
    ) -> List[RAGResponse]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions.
            top_k: Number of documents per query.
            template: Prompt template.
            
        Returns:
            List of RAGResponse objects.
        """
        responses = []
        for question in questions:
            response = self.query(question, top_k=top_k, template=template)
            responses.append(response)
        return responses
    
    def train_reft_on_conflicts(
        self,
        examples: List[Dict[str, str]],
        num_steps: int = 100,
        learning_rate: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Train ReFT intervention on conflict examples.
        
        Args:
            examples: List of dicts with query, context, answer.
            num_steps: Training steps per example.
            learning_rate: Learning rate.
            
        Returns:
            Training metrics.
        """
        if self.reft_intervention is None:
            raise RuntimeError("ReFT not enabled. Initialize with use_reft=True")
        
        from src.reft import ReFTTrainer
        
        trainer = ReFTTrainer(
            self.model,
            self.reft_intervention,
            self.tokenizer,
            target_layer=self.reft_hook.target_layer if self.reft_hook else 6,
            learning_rate=learning_rate,
            num_steps=num_steps,
        )
        
        return trainer.train(examples)
    
    def save_interventions(self, path: str) -> None:
        """Save ReFT and steering state."""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.reft_intervention:
            torch.save(
                self.reft_intervention.state_dict(),
                os.path.join(path, "reft_intervention.pt"),
            )
        
        if self.steerer and self.steerer.steering_vector is not None:
            self.steerer.save(os.path.join(path, "steering_vector.pt"))
        
        logger.info(f"Interventions saved to {path}")
    
    def load_interventions(self, path: str) -> None:
        """Load ReFT and steering state."""
        import os
        
        reft_path = os.path.join(path, "reft_intervention.pt")
        if os.path.exists(reft_path) and self.reft_intervention:
            self.reft_intervention.load_state_dict(
                torch.load(reft_path, map_location=self.device)
            )
        
        steering_path = os.path.join(path, "steering_vector.pt")
        if os.path.exists(steering_path) and self.steerer:
            self.steerer.load(steering_path)
        
        logger.info(f"Interventions loaded from {path}")
    
    def load_steering_vector(self, checkpoint_path: str) -> None:
        """
        Load a pre-computed steering vector.
        
        Args:
            checkpoint_path: Path to steering vector checkpoint (.pt file).
        """
        if self.steerer is None:
            raise RuntimeError("Steering not enabled. Initialize with use_steering=True")
        
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Steering checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'steering_vector' in checkpoint:
            steering_vector = checkpoint['steering_vector'].to(self.device)
            self.steerer.set_steering_vector(steering_vector)
            
            logger.info(f"✓ Loaded steering vector from {checkpoint_path}")
            logger.info(f"  Layer: {checkpoint.get('layer', 'unknown')}")
            logger.info(f"  Component: {checkpoint.get('component', 'unknown')}")
            logger.info(f"  Norm: {steering_vector.norm().item():.4f}")
            logger.info(f"  Examples used: {checkpoint.get('num_examples', 'unknown')}")
        else:
            raise ValueError(f"No steering_vector found in {checkpoint_path}")
    
    def load_reft_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load ReFT checkpoint trained with scripts/train_reft.py.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        if not self.reft_intervention:
            raise RuntimeError("ReFT not enabled. Initialize with use_reft=True")
        
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.reft_intervention.load_state_dict(checkpoint["intervention_state"])
        
        # Update target layer if needed
        if "target_layer" in checkpoint and self.reft_hook:
            loaded_layer = checkpoint["target_layer"]
            if loaded_layer != self.reft_hook.target_layer:
                logger.warning(
                    f"Checkpoint trained for layer {loaded_layer}, "
                    f"but pipeline uses layer {self.reft_hook.target_layer}"
                )
        
        logger.info(f"ReFT checkpoint loaded from {checkpoint_path}")


class BilingualRAGPipeline(RAGPipeline):
    """
    Bilingual RAG pipeline supporting English and Hindi.
    
    Extends base RAGPipeline with translation capabilities.
    """
    
    def __init__(
        self,
        primary_language: str = "en",
        secondary_language: str = "hi",
        translation_model: str = "Helsinki-NLP/opus-mt-en-hi",
        **kwargs,
    ) -> None:
        """
        Initialize bilingual pipeline.
        
        Args:
            primary_language: Primary language code.
            secondary_language: Secondary language code.
            translation_model: Model for translation.
            **kwargs: Arguments for base RAGPipeline.
        """
        super().__init__(**kwargs)
        
        self.primary_language = primary_language
        self.secondary_language = secondary_language
        
        # Translation models (lazy loaded)
        self._translator_to_secondary = None
        self._translator_to_primary = None
        self.translation_model_name = translation_model
    
    def _load_translators(self) -> None:
        """Lazy load translation models."""
        if self._translator_to_secondary is None:
            from transformers import MarianMTModel, MarianTokenizer
            
            # English to Hindi
            model_name = self.translation_model_name
            self._translator_to_secondary = MarianMTModel.from_pretrained(model_name)
            self._tokenizer_to_secondary = MarianTokenizer.from_pretrained(model_name)
            
            # Hindi to English (reverse)
            reverse_model = model_name.replace("en-hi", "hi-en")
            try:
                self._translator_to_primary = MarianMTModel.from_pretrained(reverse_model)
                self._tokenizer_to_primary = MarianTokenizer.from_pretrained(reverse_model)
            except Exception:
                logger.warning("Reverse translation model not available")
    
    def translate(
        self,
        text: str,
        to_language: str,
    ) -> str:
        """
        Translate text to specified language.
        
        Args:
            text: Text to translate.
            to_language: Target language code.
            
        Returns:
            Translated text.
        """
        self._load_translators()
        
        if to_language == self.secondary_language:
            model = self._translator_to_secondary
            tokenizer = self._tokenizer_to_secondary
        else:
            model = self._translator_to_primary
            tokenizer = self._tokenizer_to_primary
        
        if model is None:
            return text  # Return original if translation not available
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def query(
        self,
        question: str,
        output_language: Optional[str] = None,
        **kwargs,
    ) -> RAGResponse:
        """
        Query with optional output language translation.
        
        Args:
            question: User question.
            output_language: Language for output (None for same as input).
            **kwargs: Arguments for base query.
            
        Returns:
            RAGResponse with potentially translated answer.
        """
        # Get base response
        response = super().query(question, **kwargs)
        
        # Translate if needed
        if output_language and output_language != self.primary_language:
            response.answer = self.translate(response.answer, output_language)
            response.metadata["output_language"] = output_language
        
        return response
