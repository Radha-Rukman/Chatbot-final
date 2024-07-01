from model_loading import load_models, load_embedding_model
from llama_index import LlamaIndex, Document, SentenceSplitter

class ChatbotService:
    def __init__(self):
        self.documents = {}
        self.document_embeddings = {}
        self.tokenizer, self.model = load_models()
        self.embedding_model = load_embedding_model()
        self.index = LlamaIndex()  # Initialize LlamaIndex

    def add_document(self, doc_id, content):
        self.documents[doc_id] = content
        sentences = SentenceSplitter().split(content)
        sentence_embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
        document = Document(doc_id, sentences, sentence_embeddings)
        self.index.add_document(document)

    def answer_question(self, question, document_id=None):
        if document_id:
            if document_id not in self.documents:
                return "Document not found"
            document = self.documents[document_id]
            question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
            best_sentence = self.index.query(question_embedding)
            return self.answer_question_with_context(best_sentence, question)
        else:
            return self.answer_general_question(question)

    def answer_question_with_context(self, context, question):
        input_text = context + "\n\nQ: " + question + "\nA:"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
        outputs = self.model.generate(inputs['input_ids'], max_length=100)
        if torch.cuda.is_available():
            outputs = outputs.cpu()
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer_general_question(self, question):
        input_text = "Q: " + question + "\nA:"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
        outputs = self.model.generate(inputs['input_ids'], max_length=100)
        if torch.cuda.is_available():
            outputs = outputs.cpu()
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
