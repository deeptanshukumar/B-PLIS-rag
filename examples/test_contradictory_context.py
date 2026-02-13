"""
Test Dynamic Steering with Contradictory Context

This script tests whether dynamic steering can force the model to follow
retrieved context even when it contradicts the model's parametric knowledge.

The test creates a fake document claiming that breach of contract results in
a REWARD (opposite of reality), and checks if the model follows this fake context.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress all warnings and verbose output BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

import warnings
warnings.filterwarnings('ignore')

# Suppress logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Disable progress bars
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from src.rag_pipeline import RAGPipeline
from src.data_handler import Document


def test_contradictory_steering():
    
    print('=' * 70)
    print('Testing Dynamic Steering with Contradictory Context')
    print('=' * 70)

    print('\n  Initializing pipeline...')
    pipeline = RAGPipeline(
        model_name='google/flan-t5-base',
        use_reft=True,
        use_steering=True
    )

    print(' Loading checkpoints...')
    pipeline.load_reft_checkpoint('checkpoints/reft_flant5_full.pt')
    pipeline.load_steering_vector('checkpoints/steering_vector.pt')

    # Create a fake document with WRONG information that contradicts common knowledge
    fake_doc = Document(
        id='fake_001',
        content=(
            'According to this legal agreement, the penalty for breach of contract '
            'is a reward of 10,000 dollars paid TO the breaching party. The breaching '
            'party receives money as compensation for breaking the contract. This is '
            'standard practice in all modern contracts.'
        ),
        source='fake_contract',
        metadata={}
    )

    print(' Indexing contradictory document...')
    pipeline.index_documents([fake_doc])

    # Query that would normally have correct answer from model memory
    query = 'What is the penalty for breach of contract?'

    print(f'\n Query: {query}')
    print(f'\n What Model KNOWS (from training):')
    print('   Breach = penalties/damages paid BY breaching party')
    print(f'\n What Document SAYS (fake/wrong):')
    print('   Breach = reward of $10,000 paid TO breaching party')
    print('\n  Generating answer with dynamic steering...\n')

    # Generate response
    response = pipeline.query(query, top_k=1)

    # Display results
    print('=' * 70)
    print(f'ANSWER: {response.answer}')
    print('=' * 70)
    print(f'\n Retrieval Score: {response.scores[0]:.4f}')

    # Analyze if model followed context or memory
    answer_lower = response.answer.lower()
    
    if any(keyword in answer_lower for keyword in ['reward', 'receive', 'paid to', '10,000', '10000']):
        print('\n SUCCESS: Model followed the FAKE CONTEXT!')
        print('   Dynamic steering is working - model used contradictory document over memory.')
        return True
    elif any(keyword in answer_lower for keyword in ['penalty', 'damages', 'liable']):
        print('\n FAILURE: Model used its MEMORY instead of context')
        print('   Dynamic steering is not working - model ignored the retrieved document.')
        return False
    else:
        print('\n  UNCLEAR: Answer does not clearly show context vs memory preference')
        print(f'   Raw answer: {response.answer}')
        return None


if __name__ == '__main__':
    result = test_contradictory_steering()
    
    if result is True:
        print('\n Test passed: Dynamic steering successfully overrides model memory!')
    elif result is False:
        print('\n  Test failed: Dynamic steering did not override model memory.')
    else:
        print('\n Test inconclusive: Could not determine if steering worked.')
