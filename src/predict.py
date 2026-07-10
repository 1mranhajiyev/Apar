"""Submission faylı yaratmaq üçün CLI skript.

İstifadə:
    python src/predict.py

Tələblər:
    - data/test.csv mövcud olmalıdır
    - outputs/tfidf_pipeline.pkl mövcud olmalıdır
    - outputs/lgbm_model.pkl mövcud olmalıdır
    - outputs/label_encoder.pkl mövcud olmalıdır
    - sentence-transformers paketi quraşdırılmış olmalıdır

Yarış qaydası: inference yalnız CPU, maks 8GB RAM
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
SRC_DIR    = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from preprocess import build_features

# Ansambl ağırlıqları (notebook-da tune edilmişdir)
W_TFIDF = 0.35
W_LGBM  = 0.65


def load_model(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        print(f'❌ Tapılmadı: {path}')
        print('   Əvvəlcə notebooks/03_model.ipynb-i icra edin.')
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    print('📂 Test dataseti yüklənir...')
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test['text'] = test.apply(build_features, axis=1)
    X_test = test['text'].fillna('')

    print('🔧 Modellər yüklənir...')
    tfidf_pipeline = load_model('tfidf_pipeline.pkl')
    lgbm           = load_model('lgbm_model.pkl')
    le             = load_model('label_encoder.pkl')

    print('🤖 Transformer embedding yaradılır (CPU)...')
    embedder = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    )
    X_emb = embedder.encode(
        X_test.tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print('📊 Proqnozlar hesablanır...')
    prob_tfidf = tfidf_pipeline.predict_proba(X_test)
    prob_lgbm  = lgbm.predict_proba(X_emb)

    prob_ensemble = W_TFIDF * prob_tfidf + W_LGBM * prob_lgbm
    preds_idx = np.argmax(prob_ensemble, axis=1)
    preds = tfidf_pipeline.classes_[preds_idx]

    submission = pd.DataFrame({'id': test['id'], 'label': preds})

    # Format yoxlaması
    valid_labels = {'technical_support', 'customer_support', 'other'}
    assert set(submission['label'].unique()).issubset(valid_labels), 'Yanlış etiket!'
    assert submission['id'].nunique() == len(submission), 'Dublikat ID!'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(out_path, index=False)

    print(f'\n✅ Submission hazırdır: {out_path}')
    print(submission['label'].value_counts())


if __name__ == '__main__':
    main()
