import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python

class ResearchPaperFeatureExtractor:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        # Load scientific NLP models
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def extract_comprehensive_features(self, paper_text):
        """
        Comprehensive feature extraction for research papers
        """
        features = {
            'semantic_coherence': self._analyze_argument_coherence(paper_text),
            'technical_depth': self._compute_technical_complexity(paper_text),
            'research_clarity': self._evaluate_objective_clarity(paper_text),
            'methodology_soundness': self._assess_methodological_quality(paper_text),
            'statistical_validity': self._validate_statistical_claims(paper_text),
            'grammar_score': self._grammar_score(paper_text)
        }
        return features
    
    def _analyze_argument_coherence(self, text):
        """
        Compute semantic coherence using embedding similarity between sentences
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        embeddings = self._generate_sentence_embeddings(sentences)
        
        # Compute pairwise sentence similarities
        similarity_matrix = cosine_similarity(embeddings)
        coherence_score = np.mean(similarity_matrix)
        
        return coherence_score
    
    def _generate_sentence_embeddings(self, sentences):
        """
        Generate contextual embeddings for sentences
        """
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                sentence_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(sentence_embedding.flatten())
        
        return np.array(embeddings)
    
    def _compute_technical_complexity(self, text):
        """
        Assess technical complexity using linguistic features
        """
        doc = self.nlp(text)
        
        # Count technical terms and complex linguistic structures
        technical_term_count = sum(1 for token in doc if token.pos_ in ['NOUN', 'ADJ'] and token.is_upper)
        complex_sentence_count = sum(1 for sent in doc.sents if len(sent) > 20)
        
        technical_depth_score = (technical_term_count + complex_sentence_count) / len(list(doc.sents))
        return technical_depth_score
    
    def _evaluate_objective_clarity(self, text):
        """
        Assess clarity of research objectives
        """
        doc = self.nlp(text)
        
        # Look for explicit research objective markers
        objective_markers = ['aim', 'objective', 'goal', 'purpose']
        objective_sentences = [sent for sent in doc.sents if any(marker in sent.text.lower() for marker in objective_markers)]
        
        clarity_score = len(objective_sentences) / len(list(doc.sents))
        return clarity_score
    
    def _assess_methodological_quality(self, text):
        """
        Evaluate methodological soundness
        """
        doc = self.nlp(text)
        
        methodology_keywords = ['method', 'approach', 'methodology', 'technique']
        methodology_sentences = [sent for sent in doc.sents if any(keyword in sent.text.lower() for keyword in methodology_keywords)]
        
        method_detail_score = sum(len(sent) for sent in methodology_sentences) / len(list(doc.sents))
        return method_detail_score
    
    def _validate_statistical_claims(self, text):
        """
        Validate statistical claims and evidence
        """
        doc = self.nlp(text)
        
        statistical_terms = ['p-value', 'significance', 'statistical', 'confidence interval']
        statistical_sentences = [sent for sent in doc.sents if any(term in sent.text.lower() for term in statistical_terms)]
        
        statistical_validity_score = len(statistical_sentences) / len(list(doc.sents))
        return statistical_validity_score
    

    def _grammar_score(self, text):
        """
        Evaluate the grammar of the research paper
        """
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)  
        total_errors = len(matches)
        txt_len = len(text.split())
        error_density = total_errors/txt_len
        return error_density


