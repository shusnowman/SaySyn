from flask import Flask, jsonify, request
import random
import re
import os
from typing import List, Dict
import nltk
from nltk.corpus import stopwords

# Download Russian stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    from navec import Navec
    NAVEC_AVAILABLE = True
except ImportError:
    NAVEC_AVAILABLE = False
    print("Install navec: pip install navec")

app = Flask(__name__)

class NavecReplacer:
    def __init__(self):
        self.model = None
        self.model_path = "navec_hudlit_v1_12B_500K_300d_100q.tar"  # Fixed filename
        self.stop_words = set(stopwords.words('russian'))
        self._load_model()
    
    def _load_model(self):
        """Load Navec model"""
        if not NAVEC_AVAILABLE:
            print("Navec not installed. Install with: pip install navec")
            return
            
        try:
            if os.path.exists(self.model_path):
                print("Loading Navec model...")
                self.model = Navec.load(self.model_path)
                print(f"Model loaded! Vocabulary size: {len(self.model.vocab.words)}")
            else:
                print(f"Model file {self.model_path} not found.")
                print("Download from: https://github.com/natasha/navec")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _words_differ_significantly(self, word1: str, word2: str, min_diff: int = 2) -> bool:
        """Check if two words differ by at least min_diff characters"""
        if len(word1) != len(word2):
            return True
        
        diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
        return diff_count >= min_diff
    
    def get_similar_words(self, word: str, topn: int = 10) -> List[str]:
        """Find similar words using Navec model with filtering for significant differences"""
        if not self.model or word not in self.model:
            return []
        
        try:
            # Get word vector
            word_vector = self.model[word]
            
            # Find most similar words
            similarities = []
            for vocab_word in self.model.vocab.words:
                if (vocab_word != word and 
                    vocab_word.isalpha() and 
                    self._words_differ_significantly(word, vocab_word)):
                    try:
                        other_vector = self.model[vocab_word]
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(word_vector, other_vector)
                        similarities.append((vocab_word, similarity))
                    except:
                        continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [word for word, sim in similarities[:topn] if sim > 0.6]
            
        except Exception as e:
            print(f"Error finding similar words for '{word}': {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def extract_meaningful_words(self, phrase: str) -> List[str]:
        """Extract meaningful words using NLTK stopwords"""
        words = re.findall(r'\b[а-яё]+\b', phrase.lower())
        return [word for word in words if word not in self.stop_words and len(word) > 2]
    
    def replace_word(self, phrase: str) -> Dict:
        """Replace random word with Navec-based synonym"""
        if not self.model:
            return {
                'result': phrase,
                'replaced_word': None,
                'synonym': None,
                'error': 'Navec model not loaded. Install navec and download hudlit_12B_500K_300d_100q.tar'
            }
        
        meaningful_words = self.extract_meaningful_words(phrase)
        if not meaningful_words:
            return {
                'result': phrase,
                'replaced_word': None,
                'synonym': None,
                'error': 'No meaningful words found'
            }
        
        # Try to find synonyms for words
        for attempt in range(len(meaningful_words)):
            target_word = random.choice(meaningful_words)
            synonyms = self.get_similar_words(target_word)
            
            if synonyms:
                synonym = random.choice(synonyms)
                
                # Replace word in phrase
                pattern = r'\b' + re.escape(target_word) + r'\b'
                result = re.sub(pattern, synonym, phrase, count=1, flags=re.IGNORECASE)
                
                return {
                    'result': result,
                    'replaced_word': target_word,
                    'synonym': synonym,
                    'available_synonyms': synonyms[:3],
                    'model_info': f'Navec model with {len(self.model.vocab.words)} words'
                }
            
            meaningful_words.remove(target_word)
            if not meaningful_words:
                break
        
        return {
            'result': phrase,
            'replaced_word': None,
            'synonym': None,
            'error': 'No synonyms found in Navec model for any word'
        }

replacer = NavecReplacer()

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Словотрансформатор</title>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: #f5f7fa; }
        .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
        .status { padding: 15px; border-radius: 8px; margin-bottom: 25px; text-align: center; font-weight: 500; }
        .status-ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .input-group { margin-bottom: 20px; }
        input[type="text"] { width: 100%; padding: 16px; font-size: 16px; border: 2px solid #e9ecef; border-radius: 8px; box-sizing: border-box; transition: border-color 0.3s; }
        input[type="text"]:focus { outline: none; border-color: #3498db; }
        button { background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 16px 32px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 25px; padding: 25px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db; }
        .result-text { font-size: 18px; font-weight: 500; color: #2c3e50; margin-bottom: 15px; }
        .info { color: #6c757d; font-size: 14px; margin-bottom: 10px; }
        .synonyms { background: #e3f2fd; padding: 12px; border-radius: 6px; margin-top: 15px; }
        .error { color: #e74c3c; font-weight: 500; }
        .download-info { background: #fff3cd; color: #856404; padding: 20px; border-radius: 8px; margin-bottom: 25px; }
        .download-info a { color: #856404; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Словотрансформатор</h1>
        <div class="subtitle">Быстрая замена слов с помощью компактной векторной модели</div>
        
        <div id="modelStatus" class="status">Проверка модели...</div>
        
        <div class="download-info">
            <strong>Для работы нужна модель Navec:</strong><br>
            1. <code>pip install navec</code><br>
            2. Скачайте <code>hudlit_12B_500K_300d_100q.tar</code> с <a href="https://github.com/natasha/navec" target="_blank">GitHub Navec</a><br>
            3. Поместите файл в папку проекта
        </div>
        
        <div class="input-group">
            <input type="text" id="phrase" placeholder="Введите русскую фразу, например: семь раз отмерь, один раз отрежь" />
        </div>
        
        <button onclick="replacePhrase()">🔄 Заменить слово</button>
        
        <div id="result" style="display:none;" class="result">
            <div class="result-text" id="resultText"></div>
            <div id="info" class="info"></div>
            <div id="synonyms" class="synonyms" style="display:none;">
                <strong>Другие синонимы:</strong> <span id="synonymsList"></span>
            </div>
        </div>
    </div>

    <script>
        // Check model status
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                const statusDiv = document.getElementById('modelStatus');
                if (data.model_loaded) {
                    statusDiv.className = 'status status-ok';
                    statusDiv.textContent = `✅ Navec модель загружена (${data.vocab_size} слов)`;
                } else {
                    statusDiv.className = 'status status-error';
                    statusDiv.textContent = '❌ Модель не загружена. Установите navec и скачайте модель.';
                }
            })
            .catch(() => {
                document.getElementById('modelStatus').className = 'status status-error';
                document.getElementById('modelStatus').textContent = '❌ Ошибка подключения к серверу';
            });

        function replacePhrase() {
            const phrase = document.getElementById('phrase').value.trim();
            if (!phrase) {
                alert('Введите фразу!');
                return;
            }
            
            fetch('/replace', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({phrase: phrase})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('resultText').textContent = data.result;
                
                if (data.error) {
                    document.getElementById('info').innerHTML = 
                        `<span class="error">Ошибка: ${data.error}</span>`;
                    document.getElementById('synonyms').style.display = 'none';
                } else if (data.replaced_word && data.synonym) {
                    document.getElementById('info').innerHTML = 
                        `Заменено: <strong>"${data.replaced_word}"</strong> → <strong>"${data.synonym}"</strong>`;
                    
                    if (data.available_synonyms && data.available_synonyms.length > 0) {
                        document.getElementById('synonymsList').textContent = data.available_synonyms.join(', ');
                        document.getElementById('synonyms').style.display = 'block';
                    }
                    
                    if (data.model_info) {
                        document.getElementById('info').innerHTML += `<br><small>${data.model_info}</small>`;
                    }
                } else {
                    document.getElementById('info').textContent = 'Синонимы не найдены';
                    document.getElementById('synonyms').style.display = 'none';
                }
                
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Произошла ошибка при обращении к серверу!');
            });
        }
        
        document.getElementById('phrase').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                replacePhrase();
            }
        });
        
        // Auto-focus input
        document.getElementById('phrase').focus();
    </script>
</body>
</html>
    '''

@app.route('/status')
def status():
    model_loaded = replacer.model is not None
    vocab_size = len(replacer.model.vocab.words) if model_loaded else 0
    return jsonify({
        'model_loaded': model_loaded,
        'vocab_size': vocab_size
    })

@app.route('/replace', methods=['POST'])
def replace():
    data = request.json
    phrase = data.get('phrase', '')
    result = replacer.replace_word(phrase)
    return jsonify(result)

if __name__ == '__main__':
    print("Starting Navec Web Replacer...")
    print("Install requirements: pip install navec nltk flask")
    print("Open http://localhost:4000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=4000)