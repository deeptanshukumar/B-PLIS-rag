"""
B-PLIS-RAG: Bilingual Legal/Commerce RAG with ReFT and Activation Steering

A Retrieval-Augmented Generation system for legal domain queries with
representation fine-tuning and activation steering for enhanced faithfulness.
"""

__version__ = "0.1.0"
__author__ = "B-PLIS Team"

from src.config import Config, get_config
from src.model_loader import load_model, ModelConfig
from src.reft import ReFTIntervention, ReFTTrainer
from src.activation_steering import ActivationSteering
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.retriever import FAISSRetriever
from src.evaluator import Evaluator, EvaluationMetrics
from src.data_handler import DataHandler, LegalBenchRAG

__all__ = [
    # Config
    "Config",
    "get_config",
    # Model
    "load_model",
    "ModelConfig",
    # ReFT
    "ReFTIntervention",
    "ReFTTrainer",
    # Steering
    "ActivationSteering",
    # Pipeline
    "RAGPipeline",
    "RAGResponse",
    # Retrieval
    "FAISSRetriever",
    # Evaluation
    "Evaluator",
    "EvaluationMetrics",
    # Data
    "DataHandler",
    "LegalBenchRAG",
]
