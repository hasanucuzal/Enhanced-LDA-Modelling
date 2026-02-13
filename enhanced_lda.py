import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from docx import Document

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



doc = Document('speech.docx')
content = '\n'.join([para.text for para in doc.paragraphs])

speeches = []
lines = content.strip().split('\n')

i = 0
speech_count = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line == '#':
        speech_count += 1
        
        if i + 4 < len(lines):
            line4 = lines[i+4]
            speaker_part = ""
            keywords_part = ""
            
            if ':' in line4:
                parts = line4.split(':', 1)
                speaker_part = parts[0].strip()
                if len(parts) > 1:
                    keywords_part = parts[1].strip()
            else:
                speaker_part = line4.strip()
            
            speech = {
                'id': speech_count,
                'title': lines[i+1].strip(),
                'date': lines[i+2].strip(),
                'url': lines[i+3].strip(),
                'speaker': speaker_part,
                'keywords': keywords_part,  # KEYWORDS ADDED
                'text': ''
            }
            
            text_lines = []
            j = i + 5
            while j < len(lines) and not (lines[j].strip().startswith('#')):
                clean_line = lines[j].strip()
                text_lines.append(clean_line)
                j += 1
            
            speech['text'] = ' '.join(text_lines)
            speeches.append(speech)
            i = j
        else:
            i += 1
    else:
        i += 1

df = pd.DataFrame(speeches)

if len(df) > 0:
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    
    df.to_csv('speech.csv', index=False, encoding='utf-8')
    
    try:
        with pd.ExcelWriter('speech.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Speeches', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Speeches']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
    except Exception as e:
        print("Excel save error: {e}")
else:
    print("No speeches found!")

print(f"\n🏁 Process completed!")

df = pd.read_csv('speech.csv')

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M')

print(f"Toplam konuşma sayısı: {len(df)}")

print("\nİlk 5 konuşma:")
print(df[['date', 'title', 'word_count']].head())
print(f"\nTarih aralığı: {df['date'].min()} - {df['date'].max()}")

# 4. Temel istatistikler
print("\nKelime sayısı istatistikleri:")
print(df['word_count'].describe())

# 5. Zaman dağılımı grafiği
plt.figure(figsize=(12, 5))
df.groupby('year_month').size().plot(kind='bar', color='steelblue')
plt.title('Konuşmaların Zaman İçinde Dağılımı')
plt.xlabel('Ay')
plt.ylabel('Konuşma Sayısı')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
SMART STOPWORDS DETECTOR - OPTIMIZED VERSION
Putin Konuşmaları için Akıllı Stopwords Tespit Sistemi
Hedef: %30-40 optimal azalma oranı
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import warnings
warnings.filterwarnings('ignore')

class SmartStopwordsDetectorOptimized:
    """
    Optimize edilmiş akıllı stopwords tespit sınıfı
    Putin konuşmaları için özel olarak tasarlandı
    """
    
    def __init__(self, csv_path, text_column='text', target_reduction=35):
        """
        Parameters:
        -----------
        csv_path : str
            CSV dosya yolu
        text_column : str
            Metin sütunu adı
        target_reduction : int
            Hedeflenen azalma yüzdesi (30-40 arası optimal)
        """
        print("="*60)
        print("🤖 SMART STOPWORDS DETECTOR - OPTIMIZED")
        print("="*60)
        
        # Parametreler
        self.target_reduction = target_reduction
        self.text_column = text_column
        
        # Veriyi yükle
        print(f"\n📂 Veri yükleniyor: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"   ✓ Yüklenen satır: {len(self.df)}")
        
        # Metinleri hazırla
        self.prepare_texts()
        
        # Sonuçlar
        self.stopwords_results = {}
        self.final_stopwords = []
        
    def prepare_texts(self):
        """Metinleri temizle ve hazırla"""
        print("\n🧹 Metinler temizleniyor...")
        
        # NaN kontrolü
        self.df[self.text_column] = self.df[self.text_column].fillna('')
        
        # Gelişmiş temizleme fonksiyonu
        def advanced_clean(text):
            text = str(text).lower()
            
            # Noktalama ve özel karakterler
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Sayılar
            text = re.sub(r'\d+', '', text)
            
            # Fazla boşluklar
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Temizleme uygula
        self.df['cleaned_text'] = self.df[self.text_column].apply(advanced_clean)
        self.texts = self.df['cleaned_text'].tolist()
        
        # İstatistikler
        total_words = sum(len(text.split()) for text in self.texts)
        unique_words = len(set(' '.join(self.texts).split()))
        
        print(f"   ✓ Temizlenmiş konuşma: {len(self.texts)}")
        print(f"   ✓ Toplam kelime: {total_words:,}")
        print(f"   ✓ Benzersiz kelime: {unique_words:,}")
    
    def get_core_english_stopwords(self):
        """Çekirdek İngilizce stopwords listesi"""
        return {
            # Articles
            'a', 'an', 'the',
            
            # Common pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs',
            
            # Common prepositions
            'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'out', 'off',
            'over', 'under', 'again', 'further',
            
            # Common conjunctions
            'and', 'but', 'or', 'nor', 'so', 'yet',
            'although', 'because', 'since', 'unless',
            
            # Common verbs (to be, to have, to do)
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing',
            
            # Common adverbs
            'very', 'really', 'quite', 'just', 'only', 'also',
            'well', 'too', 'even', 'still', 'always', 'never',
            
            # Common determiners
            'this', 'that', 'these', 'those',
            'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too',
            
            # Question words
            'what', 'which', 'who', 'whom', 'whose',
            'when', 'where', 'why', 'how',
            
            # Modal verbs
            'will', 'would', 'can', 'could', 'shall', 'should',
            'may', 'might', 'must'
        }
    
    def get_putin_context_stopwords(self):
        """Putin konuşmaları bağlamına özel stopwords"""
        return {
            # Putin'e özel fiiller ve yardımcı fiiller
            'said', 'says', 'according', 'regarding', 'including',
            'within', 'without', 'upon', 'among', 'through',
            
            # Siyasi terminoloji (genel)
            'country', 'countries', 'state', 'states',
            'government', 'governments', 'people', 'peoples',
            'nation', 'national', 'international',
            
            # Zaman ifadeleri (çok genel)
            'today', 'yesterday', 'tomorrow', 'now', 'then',
            'year', 'years', 'month', 'months', 'day', 'days',
            'time', 'times', 'period', 'periods',
            
            # Miktar ifadeleri (çok genel)
            'many', 'much', 'more', 'most', 'several', 'various',
            
            # Coğrafi terimler (analize bağlı - opsiyonel)
            # 'russia', 'russian', 'ukraine', 'ukrainian',
            # 'moscow', 'kyiv', 'kiev'
        }
    
    def analyze_statistical_methods(self):
        """İstatistiksel yöntemlerle stopwords tespiti"""
        print("\n" + "="*60)
        print("📊 İSTATİSTİKSEL ANALİZ YÖNTEMLERİ")
        print("="*60)
        
        # Tüm kelimeleri say
        all_words = ' '.join(self.texts).split()
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Doküman frekansları
        doc_freq = defaultdict(int)
        for text in self.texts:
            for word in set(text.split()):
                doc_freq[word] += 1
        
        N = len(self.texts)
        
        print(f"\n📈 TEMEL İSTATİSTİKLER:")
        print(f"   Toplam kelime: {total_words:,}")
        print(f"   Benzersiz kelime: {len(word_counts):,}")
        print(f"   Ortalama doküman uzunluğu: {total_words/N:.0f} kelime")
        
        # 1. YÜKSEK FREKANSLI KELİMELER ANALİZİ
        print("\n🔍 YÜKSEK FREKANSLI KELİMELER ANALİZİ:")
        
        high_freq_words = []
        top_50 = word_counts.most_common(50)
        
        for word, freq in top_50:
            word_pct = (freq / total_words) * 100
            doc_pct = (doc_freq[word] / N) * 100
            
            # Kritik eşikler
            if word_pct > 0.1:  # %0.1'den fazla
                high_freq_words.append((word, freq, word_pct, doc_pct))
                status = "🚨 YÜKSEK" if word_pct > 0.5 else "⚠️ ORTA"
                print(f"   {status} {word:12} → %{word_pct:.2f} (doküman: %{doc_pct:.1f})")
        
        self.stopwords_results['high_freq'] = {w[0] for w in high_freq_words}
        
        # 2. YÜKSEK DOKÜMAN FREKANSI ANALİZİ
        print("\n📄 YÜKSEK DOKÜMAN FREKANSI ANALİZİ:")
        
        high_doc_words = []
        for word, df in doc_freq.items():
            doc_pct = (df / N) * 100
            if doc_pct > 50:  # %50'den fazla dokümanda
                freq = word_counts[word]
                word_pct = (freq / total_words) * 100
                high_doc_words.append((word, freq, word_pct, doc_pct))
        
        # Sırala ve göster
        high_doc_words.sort(key=lambda x: x[3], reverse=True)
        for word, freq, word_pct, doc_pct in high_doc_words[:20]:
            print(f"   📄 {word:12} → %{doc_pct:.1f} dokümanda (frekans: %{word_pct:.2f})")
        
        self.stopwords_results['high_doc'] = {w[0] for w in high_doc_words}
        
        # 3. TF-IDF ANALİZİ (DÜŞÜK SKORLU KELİMELER)
        print("\n🤖 TF-IDF ANALİZİ (Düşük Önemli Kelimeler):")
        
        tfidf_stopwords = self.tfidf_analysis()
        self.stopwords_results['low_tfidf'] = tfidf_stopwords
        
        # 4. KISA KELİMELER ANALİZİ
        print("\n🔤 KISA KELİMELER ANALİZİ:")
        short_words = {w for w in word_counts if len(w) <= 2}
        self.stopwords_results['short_words'] = short_words
        
        # Kısa kelimelerin etkisi
        short_impact = sum(word_counts[w] for w in short_words)
        print(f"   ✓ {len(short_words)} kısa kelime")
        print(f"   ✓ Toplam kullanım: {short_impact:,} (%{(short_impact/total_words)*100:.1f})")
        
        return self.stopwords_results
    
    def tfidf_analysis(self, low_percentile=20):
        """TF-IDF ile düşük önemli kelimeleri bul"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=2,
                max_df=0.85,
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(self.texts)
            features = vectorizer.get_feature_names_out()
            scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Normalize scores
            if scores.max() > scores.min():
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                norm_scores = scores
            
            # Low percentile threshold
            threshold = np.percentile(norm_scores, low_percentile)
            
            # Low TF-IDF words
            low_tfidf_words = {word for word, score in zip(features, norm_scores) 
                              if score <= threshold}
            
            # Show top low TF-IDF words
            word_scores = [(word, score) for word, score in zip(features, norm_scores) 
                          if score <= threshold]
            word_scores.sort(key=lambda x: x[1])
            
            print(f"   🎯 {len(low_tfidf_words)} düşük TF-IDF kelime")
            print(f"   📊 Eşik değeri: {threshold:.4f}")
            
            if word_scores:
                print(f"   📋 Örnekler: {', '.join([w[0] for w in word_scores[:10]])}")
            
            return low_tfidf_words
            
        except Exception as e:
            print(f"   ⚠️ TF-IDF hatası: {e}")
            return set()
    
    def ensemble_voting(self, min_votes=2):
        """Ensemble yöntemi ile birleşik stopwords belirle"""
        print("\n" + "="*60)
        print("🤝 ENSEMBLE OYLAMA SİSTEMİ")
        print("="*60)
        
        # Oylama sistemi
        votes = defaultdict(int)
        all_words_set = set()
        
        for method, words in self.stopwords_results.items():
            for word in words:
                votes[word] += 1
                all_words_set.add(word)
        
        # Oylama sonuçlarını analiz et
        print(f"\n📊 OY DAĞILIMI:")
        vote_distribution = defaultdict(int)
        for word, vote_count in votes.items():
            vote_distribution[vote_count] += 1
        
        for vote_count in sorted(vote_distribution.keys()):
            count = vote_distribution[vote_count]
            print(f"   {vote_count} yöntem tarafından seçilen: {count} kelime")
        
        # Çoğunluk oyu ile stopwords belirle
        majority_stopwords = {word for word, vote_count in votes.items() 
                             if vote_count >= min_votes}
        
        print(f"\n✅ ENSEMBLE SONUÇLARI:")
        print(f"   Toplam aday: {len(all_words_set)}")
        print(f"   Çoğunluk ({min_votes}+ oy): {len(majority_stopwords)}")
        
        return majority_stopwords
    
    def optimize_for_target(self, candidate_stopwords):
        """Hedef azalma oranına göre stopwords optimizasyonu"""
        print("\n" + "="*60)
        print(f"🎯 HEDEF OPTİMİZASYONU: %{self.target_reduction} AZALMA")
        print("="*60)
        
        # Kelime frekansları
        all_words = ' '.join(self.texts).split()
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Aday stopwords'leri frekansa göre sırala
        candidate_freq = [(w, word_counts.get(w, 0)) for w in candidate_stopwords 
                         if w in word_counts]
        candidate_freq.sort(key=lambda x: x[1], reverse=True)
        
        # Hedef frekansı hesapla
        target_freq = total_words * (self.target_reduction / 100)
        
        # Optimal stopwords'leri seç
        optimal_stopwords = []
        accumulated_freq = 0
        
        for word, freq in candidate_freq:
            if accumulated_freq + freq <= target_freq:
                optimal_stopwords.append(word)
                accumulated_freq += freq
            else:
                # Hedefe çok yakınsa ekle
                if (target_freq - accumulated_freq) / target_freq > 0.1:
                    optimal_stopwords.append(word)
                    accumulated_freq += freq
        
        # Coverage hesapla
        coverage = (accumulated_freq / total_words) * 100
        
        print(f"\n📈 OPTİMİZASYON SONUÇLARI:")
        print(f"   Başlangıç aday: {len(candidate_freq)}")
        print(f"   Seçilen stopwords: {len(optimal_stopwords)}")
        print(f"   Hedef frekans: {target_freq:,.0f}")
        print(f"   Gerçekleşen: {accumulated_freq:,.0f}")
        print(f"   KAPSAMA ORANI: %{coverage:.1f}")
        
        # İdeal aralık kontrolü
        if 25 <= coverage <= 45:
            print(f"   ✅ OPTİMAL ARALIKTA (%25-%45)")
        elif coverage < 25:
            print(f"   ⚠️ DÜŞÜK KAPSAMA, daha agresif filtreleme gerekebilir")
        else:
            print(f"   ⚠️ YÜKSEK KAPSAMA, daha seçici filtreleme gerekebilir")
        
        return optimal_stopwords
    
    def get_final_stopwords(self):
        """Nihai stopwords listesini oluştur"""
        print("\n" + "="*60)
        print("🏁 NİHAİ STOPWORDS BELİRLENİYOR")
        print("="*60)
        
        # 1. İstatistiksel analizleri çalıştır
        self.analyze_statistical_methods()
        
        # 2. Ensemble yöntemi ile adayları belirle
        candidate_stopwords = self.ensemble_voting(min_votes=2)
        
        # 3. Çekirdek stopwords ekle
        core_stopwords = self.get_core_english_stopwords()
        context_stopwords = self.get_putin_context_stopwords()
        
        # 4. Birleştir
        all_candidates = candidate_stopwords | core_stopwords | context_stopwords
        
        # 5. Hedef optimizasyonu
        self.final_stopwords = self.optimize_for_target(all_candidates)
        
        # 6. İstatistikleri göster
        self.show_final_statistics()
        
        return self.final_stopwords
    
    def show_final_statistics(self):
        """Nihai istatistikleri göster"""
        # Kelime frekansları
        all_words = ' '.join(self.texts).split()
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Stopwords istatistikleri
        stopwords_set = set(self.final_stopwords)
        stopwords_freq = sum(word_counts.get(w, 0) for w in stopwords_set)
        coverage = (stopwords_freq / total_words) * 100
        
        print(f"\n📊 NİHAİ İSTATİSTİKLER:")
        print(f"   Stopwords sayısı: {len(self.final_stopwords)}")
        print(f"   Kapsanan kelime: {stopwords_freq:,}")
        print(f"   Toplam kelime: {total_words:,}")
        print(f"   KAPSAMA ORANI: %{coverage:.1f}")
        
        print(f"\n🏆 EN ETKİLİ 25 STOPWORDS:")
        print("-" * 60)
        
        # Stopwords'leri frekansa göre sırala
        stopword_stats = []
        for word in self.final_stopwords:
            freq = word_counts.get(word, 0)
            if freq > 0:
                pct = (freq / total_words) * 100
                stopword_stats.append((word, freq, pct))
        
        stopword_stats.sort(key=lambda x: x[1], reverse=True)
        
        for i, (word, freq, pct) in enumerate(stopword_stats[:25], 1):
            doc_count = sum(1 for text in self.texts if word in text)
            doc_pct = (doc_count / len(self.texts)) * 100
            print(f"{i:2}. {word:15} → {freq:6,} kez (%{pct:.2f}) | %{doc_pct:.0f} dokümanda")
    
    def apply_stopwords_filter(self):
        """Stopwords'leri uygula ve sonuçları göster"""
        print("\n" + "="*60)
        print("🔄 STOPWORDS FİLTRELEME UYGULANIYOR")
        print("="*60)
        
        stopwords_set = set(self.final_stopwords)
        
        # Filtreleme fonksiyonu
        def filter_text(text):
            words = text.split()
            filtered = [w for w in words if w not in stopwords_set]
            return ' '.join(filtered)
        
        # Uygula
        self.df['filtered_text'] = self.df['cleaned_text'].apply(filter_text)
        
        # İstatistikler hesapla
        original_counts = [len(t.split()) for t in self.texts]
        filtered_counts = [len(t.split()) for t in self.df['filtered_text']]
        
        original_total = sum(original_counts)
        filtered_total = sum(filtered_counts)
        reduction_pct = ((original_total - filtered_total) / original_total) * 100
        
        print(f"\n📈 FİLTRELEME SONUÇLARI:")
        print(f"   Orijinal toplam: {original_total:,} kelime")
        print(f"   Filtrelenmiş: {filtered_total:,} kelime")
        print(f"   Çıkarılan: {original_total - filtered_total:,} kelime")
        print(f"   AZALMA ORANI: %{reduction_pct:.1f}")
        
        # Örnek karşılaştırma
        print(f"\n🔍 ÖRNEK KARŞILAŞTIRMA:")
        sample_idx = min(2, len(self.df) - 1)
        
        original_text = self.texts[sample_idx]
        filtered_text = self.df['filtered_text'].iloc[sample_idx]
        
        print(f"   Konuşma #{sample_idx + 1}:")
        print(f"   Orijinal: {len(original_text.split())} kelime")
        print(f"   Filtrelenmiş: {len(filtered_text.split())} kelime")
        print(f"   Çıkarılan: {len(original_text.split()) - len(filtered_text.split())} kelime")
        
        print(f"\n   Filtrelenmiş metin (ilk 250 karakter):")
        print(f"   \"{filtered_text[:250]}...\"")
        
        return reduction_pct
    
    def save_results(self, output_dir='smart_stopwords_results'):
        """Sonuçları kaydet"""
        print("\n" + "="*60)
        print("💾 SONUÇLAR KAYDEDİLİYOR")
        print("="*60)
        
        # Klasör oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Stopwords listesini kaydet
        stopwords_file = f'{output_dir}/smart_stopwords.txt'
        with open(stopwords_file, 'w', encoding='utf-8') as f:
            f.write("# SMART STOPWORDS DETECTOR - OPTIMIZED RESULTS\n")
            f.write(f"# Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Toplam konuşma: {len(self.texts)}\n")
            f.write(f"# Stopwords sayısı: {len(self.final_stopwords)}\n")
            f.write(f"# Hedef azalma: %{self.target_reduction}\n\n")
            
            # Kelime frekansları
            all_words = ' '.join(self.texts).split()
            word_counts = Counter(all_words)
            
            f.write("KELİME | FREKANS | YÜZDE | DOKÜMAN_YÜZDESİ\n")
            f.write("-"*50 + "\n")
            
            for word in self.final_stopwords:
                freq = word_counts.get(word, 0)
                word_pct = (freq / len(all_words)) * 100 if len(all_words) > 0 else 0
                doc_count = sum(1 for text in self.texts if word in text)
                doc_pct = (doc_count / len(self.texts)) * 100
                
                f.write(f"{word:<20} | {freq:>8,} | %{word_pct:>5.2f} | %{doc_pct:>5.1f}\n")
        
        # 2. Filtrelenmiş veriyi kaydet
        filtered_file = f'{output_dir}/filtered_speeches.csv'
        self.df.to_csv(filtered_file, index=False, encoding='utf-8')
        
        # 3. Analiz raporu oluştur
        report_file = f'{output_dir}/analysis_report.txt'
        self.create_analysis_report(report_file)
        
        print(f"\n✅ KAYDEDİLEN DOSYALAR:")
        print(f"   📄 {stopwords_file}")
        print(f"   📄 {filtered_file}")
        print(f"   📄 {report_file}")
        print(f"\n   📁 Tüm sonuçlar: {os.path.abspath(output_dir)}/")
    
    def create_analysis_report(self, report_file):
        """Analiz raporu oluştur"""
        all_words = ' '.join(self.texts).split()
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SMART STOPWORDS DETECTOR - ANALİZ RAPORU\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"ANALİZ TARİHİ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"VERİ SETİ: {len(self.texts)} konuşma\n")
            f.write(f"TOPLAM KELİME: {total_words:,}\n")
            f.write(f"BENZERSİZ KELİME: {len(word_counts):,}\n")
            f.write(f"HEDEF AZALMA: %{self.target_reduction}\n\n")
            
            f.write("\nSTOPWORDS ANALİZİ:\n")
            f.write("-"*40 + "\n")
            
            for method, words in self.stopwords_results.items():
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Kelime sayısı: {len(words)}\n")
                
                # İlk 10 kelime
                sorted_words = sorted(words, key=lambda x: word_counts.get(x, 0), reverse=True)
                f.write(f"  Örnekler: {', '.join(list(sorted_words)[:10])}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("NİHAİ STOPWORDS İSTATİSTİKLERİ:\n")
            f.write("-"*40 + "\n")
            
            stopwords_set = set(self.final_stopwords)
            stopwords_freq = sum(word_counts.get(w, 0) for w in stopwords_set)
            coverage = (stopwords_freq / total_words) * 100
            
            f.write(f"Toplam stopwords: {len(self.final_stopwords)}\n")
            f.write(f"Kapsanan kelime: {stopwords_freq:,}\n")
            f.write(f"Kapsama oranı: %{coverage:.1f}\n\n")
            
            f.write("EN ETKİLİ 50 STOPWORDS:\n")
            f.write("-"*40 + "\n")
            
            stopword_stats = []
            for word in self.final_stopwords:
                freq = word_counts.get(word, 0)
                if freq > 0:
                    pct = (freq / total_words) * 100
                    stopword_stats.append((word, freq, pct))
            
            stopword_stats.sort(key=lambda x: x[1], reverse=True)
            
            for i, (word, freq, pct) in enumerate(stopword_stats[:50], 1):
                doc_count = sum(1 for text in self.texts if word in text)
                doc_pct = (doc_count / len(self.texts)) * 100
                f.write(f"{i:3}. {word:<20} {freq:>8,} kez (%{pct:>5.2f}) | %{doc_pct:>5.1f} doküman\n")
    
    def visualize_results(self):
        """Sonuçları görselleştir"""
        print("\n" + "="*60)
        print("📊 GÖRSELLEŞTİRME OLUŞTURULUYOR")
        print("="*60)
        
        try:
            # 1. Stopwords dağılımı
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Smart Stopwords Detector - Analiz Sonuçları', 
                        fontsize=16, fontweight='bold')
            
            # Veri hazırla
            all_words = ' '.join(self.texts).split()
            word_counts = Counter(all_words)
            stopwords_set = set(self.final_stopwords)
            
            # A. Stopwords frekans dağılımı
            ax1 = axes[0, 0]
            top_stopwords = sorted(stopwords_set, 
                                  key=lambda x: word_counts.get(x, 0), 
                                  reverse=True)[:15]
            top_freqs = [word_counts.get(w, 0) for w in top_stopwords]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_stopwords)))
            ax1.barh(range(len(top_stopwords)), top_freqs, color=colors)
            ax1.set_yticks(range(len(top_stopwords)))
            ax1.set_yticklabels(top_stopwords)
            ax1.set_xlabel('Frekans')
            ax1.set_title('En Sık Kullanılan 15 Stopwords')
            ax1.invert_yaxis()
            
            # B. Yöntemlere göre stopwords sayısı
            ax2 = axes[0, 1]
            methods = list(self.stopwords_results.keys())
            method_counts = [len(words) for words in self.stopwords_results.values()]
            
            bars = ax2.bar(methods, method_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax2.set_title('Yöntemlere Göre Stopwords Sayısı')
            ax2.set_ylabel('Kelime Sayısı')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom')
            
            # C. Filtreleme etkisi
            ax3 = axes[1, 0]
            original_counts = [len(t.split()) for t in self.texts[:5]]
            filtered_counts = [len(t.split()) for t in self.df['filtered_text'][:5]]
            
            x = range(len(original_counts))
            width = 0.35
            ax3.bar([i - width/2 for i in x], original_counts, width, 
                   label='Orijinal', color='gray', alpha=0.7)
            ax3.bar([i + width/2 for i in x], filtered_counts, width, 
                   label='Filtrelenmiş', color='lightblue', alpha=0.7)
            ax3.set_title('İlk 5 Konuşma - Filtreleme Etkisi')
            ax3.set_xlabel('Konuşma No')
            ax3.set_ylabel('Kelime Sayısı')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'#{i+1}' for i in x])
            ax3.legend()
            
            # D. Stopwords uzunluk dağılımı
            ax4 = axes[1, 1]
            stopword_lengths = [len(w) for w in stopwords_set]
            length_counts = Counter(stopword_lengths)
            
            lengths = sorted(length_counts.keys())
            counts = [length_counts[l] for l in lengths]
            
            ax4.bar(lengths, counts, color='darkorange', alpha=0.7)
            ax4.set_title('Stopwords Uzunluk Dağılımı')
            ax4.set_xlabel('Kelime Uzunluğu')
            ax4.set_ylabel('Kelime Sayısı')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Görselleştirmeler oluşturuldu")
            
        except Exception as e:
            print(f"⚠️ Görselleştirme hatası: {e}")
    
    def run_full_analysis(self):
        """Tam analiz pipeline'ını çalıştır"""
        print("\n" + "="*60)
        print("🚀 TAM ANALİZ PIPELINE BAŞLATILIYOR")
        print("="*60)
        
        try:
            # 1. Stopwords belirle
            stopwords = self.get_final_stopwords()
            
            # 2. Filtrelemeyi uygula
            reduction = self.apply_stopwords_filter()
            
            # 3. Görselleştir
            self.visualize_results()
            
            # 4. Kaydet
            self.save_results()
            
            # 5. Sonuç özeti
            print("\n" + "="*60)
            print("🎉 ANALİZ BAŞARIYLA TAMAMLANDI!")
            print("="*60)
            print(f"✓ Toplam konuşma: {len(self.texts)}")
            print(f"✓ Stopwords sayısı: {len(stopwords)}")
            print(f"✓ Azalma oranı: %{reduction:.1f}")
            print(f"✓ Hedef azalma: %{self.target_reduction}")
            
            if abs(reduction - self.target_reduction) <= 10:
                print(f"✓ ✅ HEDEFE YAKIN (fark: %{abs(reduction - self.target_reduction):.1f})")
            else:
                print(f"✓ ⚠️ HEDEFTEN UZAK (fark: %{abs(reduction - self.target_reduction):.1f})")
            
            print(f"\n📁 Sonuçlar: 'smart_stopwords_results/' klasöründe")
            
            return {
                'stopwords': stopwords,
                'reduction': reduction,
                'target': self.target_reduction,
                'success': abs(reduction - self.target_reduction) <= 10
            }
            
        except Exception as e:
            print(f"❌ ANALİZ HATASI: {e}")
            return None


# ============================================================================
# ÇALIŞTIRMA FONKSİYONU
# ============================================================================

def run_smart_stopwords_detection(csv_path, target_reduction=35):
 
    print("\n" + "="*60)
    print("🚀 SMART STOPWORDS DETECTOR - PUTİN KONUŞMALARI")
    print("="*60)
    
    # Parametre kontrolü
    if not 20 <= target_reduction <= 50:
        return
    
    try:
        # Detector'ı başlat
        detector = SmartStopwordsDetectorOptimized(
            csv_path=csv_path,
            text_column='text',
            target_reduction=target_reduction
        )
        
        # Tam analizi çalıştır
        results = detector.run_full_analysis()
         
    except FileNotFoundError:
        print(f"\n❌ HATA: '{csv_path}' dosyası bulunamadı!")
        print("   Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"\n❌ HATA: {e}")


if __name__ == "__main__":


    CSV_FILE = "speech.csv"
    TARGET_REDUCTION = 35 
    
    run_smart_stopwords_detection(CSV_FILE, TARGET_REDUCTION)

# -*- coding: utf-8 -*-
"""
PUTİN KONUŞMALARI - GELİŞTİRİLMİŞ FİNAL LDA ANALİZİ
Zaman analizi, güven aralıkları grafikleri ve önemli olay işaretleyicileri eklendi
Konu ayrıştırma iyileştirildi - STOPWORDS GÜNCELLENDİ
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from collections import Counter
from datetime import datetime
import random
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class PutinLDAGelistirilmis:
    """Geliştirilmiş Putin LDA analizi"""
    
    def __init__(self, csv_path, random_seed=42, n_topics=None):
        print("="*70)
        print("🔥 PUTİN KONUŞMALARI - GELİŞTİRİLMİŞ LDA ANALİZİ")
        print("="*70)
        
        # Tüm random seed'leri sabitle
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        self.df = pd.read_csv(csv_path)
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['year_month'] = self.df['date'].dt.to_period('M')
            self.df['year_quarter'] = self.df['date'].dt.to_period('Q')
        
        print(f"✓ {len(self.df)} konuşma yüklendi")
        if 'date' in self.df.columns:
            print(f"✓ Tarih aralığı: {self.df['date'].min().strftime('%Y-%m')} - {self.df['date'].max().strftime('%Y-%m')}")
        
        self.prepare_texts_gelistirilmis()
        self.n_topics = n_topics
    
    def prepare_texts_gelistirilmis(self):
        """Geliştirilmiş metin temizleme - GÜNCELLENMİŞ STOPWORDS"""
        print("\n🧹 METİNLER GELİŞTİRİLMİŞ TEMİZLENİYOR...")
        
        if 'filtered_text' in self.df.columns:
            raw_texts = self.df['filtered_text'].fillna('').tolist()
        else:
            raw_texts = self.df['text'].fillna('').tolist()
        
        # GÜNCELLENMİŞ stopwords listesi - TÜM VERİLEN STOPWORDS EKLENDİ
        GUNCELLENMIS_STOPWORDS = set([
            # Pronouns
            'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'there', 'these',
            
            # Common verbs
            'be', 'is', 'am', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'say', 'says', 'said',
            
            # Quantifiers
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'how', 'why',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'will', 'just', 'should', 'now',
            
            # Çok genel kelimeler
            'question', 'year', 'then', 'yes', 'no',
            'many', 'much', 'also', 'very', 'really',
            'well', 'good', 'better', 'best',
            
            # Putin'e özel
            'putin', 'vladimir', 'president',
            'thank', 'thanks', 'thank you', 'ladies gentlemen',
            'ladies', 'gentlemen',
            
            # Para birimleri ve miktarlar
            'million tonnes', 'billion rubles', 'million dollars', 'billion dollars',
            'trillion rubles', 'thousand tonnes',
            
            # Zaman ifadeleri
            'three years', 'eight years', 'past years', 'recent years',
            'next years', 'coming years', 'previous years',
            
            # Genel miktar ifadeleri
            'large scale', 'small scale', 'high level', 'low level',
            'great deal', 'first time', 'last time',
            
            # Karşılaştırmalar
            'compared with', 'compared to', 'in comparison',
            
            # Tek kelime olarak da ekle
            'million', 'billion', 'trillion', 'thousand',
            'tonnes', 'rubles', 'dollars', 'euros',
            'years', 'months', 'weeks', 'days', 'talking', 'about', 'first', 'point',
            'between', 'think', 'about', 'during', 'dmitry', 'peskov', 'pavel', 'zarubin',
            'would', 'like', 'would like', 'russian', 'federation', 'into', 'account', 'comrade',
            'saudi', 'arabia', 'please', 'ahead', 'alexander', 'lukashenko', 'families', 'children',
            'fyodor', 'lukyanov', 'over','past', 'people', 'republic','prime', 'minister',
            'long', 'term','time',
            'afternoon', 'name','lvova', 'belova', 'kherson','konstantin', 'panyushkin',
            'proceed','from','around','world',
            'commander','chief','extremely','important','large','scale','small','medium','sized',
            'among','things','make','sure','minimum','wage',
            'medical', 'check', 'everything','done', 'continue','work','took','part','even' ,'though',
            'arab', 'emirates','percent','business','countries','taking', 'place', 'without', 'doubt','want','emphasise'

        ])
        
        def gelistirilmis_clean_text(text):
            text = str(text).lower()
            
            # Noktalama ve sayıları kaldır
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            
            # Kelimelere ayır
            words = text.split()
            
            # Geliştirilmiş filtreleme
            filtered_words = []
            for w in words:
                if len(w) < 4:  # 4 karakterden kısa kelimeleri filtrele
                    continue
                if w in GUNCELLENMIS_STOPWORDS:
                    continue
                filtered_words.append(w)
            
            cleaned_text = ' '.join(filtered_words)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            return cleaned_text
        
        self.texts = []
        for i, t in enumerate(raw_texts):
            cleaned = gelistirilmis_clean_text(t)
            self.texts.append(cleaned)
        
        total_words = sum(len(t.split()) for t in self.texts)
        unique_words = len(set(' '.join(self.texts).split()))
        
        print(f"✓ Toplam kelime: {total_words:,}")
        print(f"✓ Benzersiz kelime: {unique_words:,}")
        print(f"✓ Ortalama kelime/konuşma: {total_words/len(self.texts):.0f}")
        
        return self.texts
    
    def create_dtm_gelistirilmis(self, n_topics=5):
        """Geliştirilmiş DTM - daha iyi konu ayrışması için"""
        print("\n📊 GELİŞTİRİLMİŞ DOCUMENT-TERM MATRIX OLUŞTURULUYOR...")
        
        # KONU AYRIŞMASI İÇİN OPTİMİZE PARAMETRELER
        if n_topics == 5:
            max_features = 1200  # Artırıldı - daha fazla özellik
            min_df = 3  # Azaltıldı - daha fazla n-gram
            max_df = 0.5  # Azaltıldı - çok yaygın terimleri filtrele
            ngram_range = (2, 4)  # Genişletildi - 4-gram'a kadar
        else:
            max_features = 1000
            min_df = 4
            max_df = 0.6
            ngram_range = (2, 3)
        
        print(f"  → N-gram: {ngram_range[0]}-{ngram_range[1]}")
        print(f"  → Max özellik: {max_features}")
        print(f"  → Min DF: {min_df}")
        print(f"  → Max DF: %{max_df*100:.0f}")
        print(f"  → Random seed: {self.random_seed}")
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=None,  # Stopwords'leri kendimiz filtreledik
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{4,}\b',  # 4 karakterden azları filtrele
        )
        
        self.dtm = self.vectorizer.fit_transform(self.texts)
        
        print(f"✓ Belge sayısı: {self.dtm.shape[0]}")
        print(f"✓ Terim sayısı: {self.dtm.shape[1]}")
        
        # En sık terimleri göster
        feature_names = self.vectorizer.get_feature_names_out()
        term_freq = np.asarray(self.dtm.sum(axis=0)).flatten()
        top_indices = term_freq.argsort()[-15:][::-1]
        
        print(f"\n🔝 EN SIK 15 N-GRAM:")
        print("-" * 50)
        for idx in top_indices[:15]:
            term = feature_names[idx]
            freq = term_freq[idx]
            percentage = (freq / self.dtm.sum()) * 100
            print(f"  {term:40} → {freq:6,} kez (%{percentage:.2f})")
        
        return self.dtm
    
    def perform_lda_gelistirilmis(self, n_topics=5):
        """Geliştirilmiş LDA - daha iyi konu ayrışması"""
        print(f"\n" + "="*70)
        print(f"🧠 GELİŞTİRİLMİŞ LDA ANALİZİ ({n_topics} KONU)")
        print("="*70)
        
        # GELİŞTİRİLMİŞ LDA PARAMETRELERİ - daha iyi ayrışma için
        print(f"\n🎯 GELİŞTİRİLMİŞ LDA PARAMETRELERİ:")
        print(f"  • Konu sayısı: {n_topics}")
        print(f"  • Random seed: {self.random_seed}")
        print(f"  • Max iterasyon: 50")  # Artırıldı
        print(f"  • Doc-topic prior: 0.1")  # Azaltıldı - daha sıkı dağılım
        print(f"  • Topic-word prior: 0.01")  # Azaltıldı - daha spesifik kelimeler
        print(f"  • Learning decay: 0.7")  # Daha yavaş öğrenme
        
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_seed,
            learning_method='online',
            max_iter=50,  # Artırıldı
            learning_offset=10.0,
            learning_decay=0.7,  # Optimize edildi
            doc_topic_prior=0.1,  # Azaltıldı
            topic_word_prior=0.01,  # Azaltıldı
            n_jobs=-1,
            verbose=1
        )
        
        print("\n📚 LDA modeli eğitiliyor...")
        self.lda.fit(self.dtm)
        
        print(f"\n✓ Model eğitimi tamamlandı")
        print(f"✓ Final perplexity: {self.lda.perplexity(self.dtm):.1f}")
        
        # Konuları çıkar
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\n🎯 {n_topics} KONU BULUNDU:")
        print("="*70)
        
        self.topics = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            # Her konu için top 15 n-gram (artırıldı)
            top_indices = topic.argsort()[-15:][::-1]
            top_ngrams = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            
            # GELİŞTİRİLMİŞ konu etiketleme
            topic_label = self.interpret_topic_gelistirilmis(top_ngrams, topic_idx, n_topics)
            
            self.topics.append({
                'id': topic_idx,
                'label': topic_label,
                'keywords': top_ngrams,
                'weights': top_weights,
                'top_keywords': top_ngrams[:6]
            })
            
            print(f"\n📌 KONU {topic_idx + 1}: {topic_label}")
            print("-" * 50)
            
            print("  🔑 ÖNEMLİ N-GRAM'LAR:")
            for i in range(0, min(15, len(top_ngrams)), 5):
                chunk = top_ngrams[i:i+5]
                if chunk:
                    print(f"     • {', '.join(chunk)}")
        
        # Doküman-konu dağılımı
        self.topic_distribution = self.lda.transform(self.dtm)
        self.df['dominant_topic'] = self.topic_distribution.argmax(axis=1)
        self.df['topic_confidence'] = self.topic_distribution.max(axis=1)
        
        # Konu güven aralıklarını hesapla
        self.calculate_topic_confidence_intervals()
        
        return self.lda
    
    def calculate_topic_confidence_intervals(self):
        """Konu bazında güven aralıklarını hesapla"""
        print("\n📊 KONU BAZINDA GÜVEN ARALIKLARI HESAPLANIYOR...")
        
        self.topic_confidence_stats = []
        
        for topic in self.topics:
            topic_id = topic['id']
            topic_confidences = self.df[self.df['dominant_topic'] == topic_id]['topic_confidence']
            
            if len(topic_confidences) > 0:
                mean_confidence = topic_confidences.mean()
                std_confidence = topic_confidences.std()
                n_samples = len(topic_confidences)
                
                # %95 güven aralığı
                if n_samples > 1:
                    # t-dağılımı kullan
                    t_value = stats.t.ppf(0.975, n_samples - 1)
                    margin_of_error = t_value * (std_confidence / np.sqrt(n_samples))
                    ci_lower = mean_confidence - margin_of_error
                    ci_upper = mean_confidence + margin_of_error
                else:
                    ci_lower = mean_confidence
                    ci_upper = mean_confidence
                
                # Konfidans seviyesi kategorisi
                if mean_confidence >= 0.8:
                    confidence_level = "ÇOK YÜKSEK"
                    level_color = "🟢"
                elif mean_confidence >= 0.6:
                    confidence_level = "YÜKSEK"
                    level_color = "🟡"
                elif mean_confidence >= 0.4:
                    confidence_level = "ORTA"
                    level_color = "🟠"
                else:
                    confidence_level = "DÜŞÜK"
                    level_color = "🔴"
                
                self.topic_confidence_stats.append({
                    'topic_id': topic_id,
                    'topic_label': topic['label'],
                    'n_documents': n_samples,
                    'mean_confidence': mean_confidence,
                    'std_confidence': std_confidence,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'margin_of_error': margin_of_error if n_samples > 1 else 0,
                    'confidence_level': confidence_level,
                    'level_color': level_color
                })
                
                print(f"  • Konu {topic_id+1}: {topic['label'][:40]}...")
                print(f"    → Ortalama güven: {mean_confidence:.3f} ± {margin_of_error:.3f}")
                print(f"    → %95 GA: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"    → Seviye: {confidence_level} {level_color}")
        
        return self.topic_confidence_stats
    
    def interpret_topic_gelistirilmis(self, ngrams, topic_id, n_topics):
        """Geliştirilmiş konu etiketleme - BENZERSİZ etiketler için"""
        
        # Tüm n-gram'ları analiz et
        ngrams_text = ' '.join(ngrams).lower()
        
        # ÇOK SPESİFİK KONTROLLER - benzersiz etiketler için
        if 'kiev regime' in ngrams_text and 'donetsk' in ngrams_text:
            return 'UKRAYNA: KIEV REJİMİ VE DONBAS BAĞIMSIZLIĞI'
        
        elif 'terrorist attack' in ngrams_text and 'crimean bridge' in ngrams_text:
            return 'TERÖR SALDIRILARI: KRİM KÖPRÜSÜ GÜVENLİĞİ'
        
        elif 'siege leningrad' in ngrams_text:
            return 'TARİHSEL ANILAR: LENİNGRAD KUŞATMASI'
        
        elif 'great patriotic war' in ngrams_text:
            return 'TARİHSEL MİRAS: BÜYÜK VATANSEVERLİK SAVAŞI'
        
        elif 'nuclear weapons' in ngrams_text:
            return 'STRATEJİK SAVUNMA: NÜKLEER SİLAH SİSTEMLERİ'
        
        elif 'artificial intelligence' in ngrams_text:
            return 'TEKNOLOJİK DEVRİM: YAPAY ZEKA GELİŞİMİ'
        
        elif 'middle east' in ngrams_text:
            return 'DIŞ POLİTİKA: ORTA DOĞU DİPLOMASİSİ'
        
        elif 'south africa' in ngrams_text:
            return 'ULUSLARARASI İLİŞKİLER: AFRİKA İŞBİRLİĞİ'
        
        elif 'energy resources' in ngrams_text:
            return 'EKONOMİK POLİTİKA: ENERJİ KAYNAKLARI'
        
        elif 'economic sanctions' in ngrams_text:
            return 'EKONOMİK MÜCADELE: YAPTIRIMLAR VE FİNANS'
        
        elif 'special military operation' in ngrams_text:
            return 'ASKERİ STRATEJİ: ÖZEL ASKERİ OPERASYON'
        
        # Kelime frekanslarına göre benzersiz etiket
        word_counter = Counter()
        for ngram in ngrams[:10]:
            words = ngram.split()
            for word in words:
                word = word.lower()
                if len(word) > 5:  # Uzun kelimeleri tercih et
                    word_counter[word] += 1
        
        # En sık 3 benzersiz kelime
        top_words = []
        seen_words = set()
        for word, count in word_counter.most_common(10):
            if word not in seen_words and len(top_words) < 3:
                top_words.append(word.upper())
                seen_words.add(word)
        
        # Konu ID'sine göre özel etiketler (n_topics'e göre)
        if n_topics == 5:
            specific_labels = {
                0: 'UKRAYNA SAVAŞI: ASKERİ OPERASYONLAR VE STRATEJİ',
                1: 'TARİHSEL HAFIZA: SAVAŞ ANILARI VE MİLLİ KİMLİK',
                2: 'DEVLET YÖNETİMİ: İÇ POLİTİKA VE KURUMSAL REFORMLAR',
                3: 'EKONOMİK KALKINMA: TEKNOLOJİ VE SANAYİ POLİTİKASI',
                4: 'ULUSLARARASI DİPLOMASİ: BÖLGESEL İŞBİRLİKLERİ'
            }
            if topic_id in specific_labels:
                return specific_labels[topic_id]
        
        # Benzersiz etiket oluştur
        if len(top_words) >= 2:
            return f"{top_words[0]} VE {top_words[1]} POLİTİKALARI"
        
        return f"KONU {topic_id+1}: SİYASİ ANALİZ"
    
    def zaman_analizi_grafikleri(self):
        """Zaman analizi grafikleri - ALT ALTA DÜZEN ve ÖNEMLİ OLAYLAR EKLENDİ"""
        print("\n⏰ ZAMAN ANALİZİ GRAFİKLERİ OLUŞTURULUYOR (ÖNEMLİ OLAYLAR EKLENDİ)...")
        
        if 'date' not in self.df.columns:
            print("⚠️  Tarih verisi yok, zaman analizi yapılamıyor")
            return
        
        # Aylık konu dağılımı
        self.df['year_month_dt'] = self.df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Yarıyıl (yarım yıl) hesapla
        self.df['half_year'] = self.df['date'].dt.year.astype(str) + '-' + self.df['date'].dt.quarter.apply(
            lambda q: 'H1' if q <= 2 else 'H2'
        )
        
        # Konu etiketlerini DataFrame'e ekle
        topic_labels = {t['id']: t['label'] for t in self.topics}
        self.df['topic_label'] = self.df['dominant_topic'].map(topic_labels)
        
        # Zaman serisi analizi
        monthly_data = self.df.groupby(['year_month_dt', 'topic_label']).size().unstack(fill_value=0)
        
        # GRAFİK 1: Zaman içinde konu dağılımı (alan grafiği) - TEK BAŞINA
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(monthly_data.columns)))
        
        monthly_data.plot.area(ax=ax1, alpha=0.8, color=colors)
        
        # ÖNEMLİ OLAYLARI EKLE - Dikey çizgiler
        olaylar = [
            {'tarih': '2022-02-24', 'etiket': 'Ukrayna İşgali', 'renk': 'red', 'alpha': 0.7},
            {'tarih': '2022-03-12', 'etiket': 'SWIFT Yaptırımları', 'renk': 'orange', 'alpha': 0.7},
            {'tarih': '2022-10-08', 'etiket': 'Kırım Köprüsü Patlaması', 'renk': 'darkred', 'alpha': 0.7},
            {'tarih': '2022-11-11', 'etiket': 'Herson Geri Alındı', 'renk': 'green', 'alpha': 0.7},
            {'tarih': '2023-06-23', 'etiket': 'Wagner İsyanı', 'renk': 'purple', 'alpha': 0.7},
            {'tarih': '2023-04-04', 'etiket': 'Finlandiya NATO', 'renk': 'blue', 'alpha': 0.7},
            {'tarih': '2025-02-28', 'etiket': 'Trump-Zelenski Krizi', 'renk': 'brown', 'alpha': 0.7},
            {'tarih': '2025-08-15', 'etiket': 'Trump-Putin Alaska', 'renk': 'cyan', 'alpha': 0.7},
            {'tarih': '2025-11-21', 'etiket': 'Trump Barış Planı', 'renk': 'magenta', 'alpha': 0.7}
        ]
        
        for olay in olaylar:
            olay_tarih = pd.to_datetime(olay['tarih'])
            if ax1.get_xlim()[0] <= mdates.date2num(olay_tarih) <= ax1.get_xlim()[1]:
                ax1.axvline(x=olay_tarih, color=olay['renk'], linestyle='--', 
                          alpha=olay['alpha'], linewidth=2)
                # Etiketi ekle
                ax1.text(olay_tarih, ax1.get_ylim()[1]*0.95, olay['etiket'], 
                       rotation=90, verticalalignment='top',
                       color=olay['renk'], fontsize=8, fontweight='bold',
                       alpha=0.8)
        
        ax1.set_title(f'Putin Konuşmaları - Zaman İçinde Konu Dağılımı (Seed: {self.random_seed})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tarih')
        ax1.set_ylabel('Konuşma Sayısı')
        ax1.legend(title='Konular', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Tarih formatı
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()
        
        # GRAFİK 2: YARIM YILLIK KONU DAĞILIMI - TEK BAŞINA
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        
        # Yarıyıl (yarım yıl) bazında konu dağılımı
        half_year_data = self.df.groupby(['half_year', 'topic_label']).size().unstack(fill_value=0)
        
        # Yarıyılları sırala
        half_year_data = half_year_data.sort_index()
        
        # Son 8 yarıyıl
        last_8_half_years = half_year_data.index[-8:] if len(half_year_data) > 8 else half_year_data.index
        
        half_year_data.loc[last_8_half_years].plot(kind='bar', stacked=True, ax=ax2, 
                                                   alpha=0.85, color=colors)
        
        # Önemli olayların olduğu yarıyılları vurgula
        olay_cizelgesi = {
            '2022-H1': 'Ukrayna İşgali',
            '2022-H1': 'SWIFT Yaptırımları',
            '2022-H2': 'Kırım Köprüsü',
            '2022-H2': 'Herson Kurtarıldı',
            '2023-H1': 'Finlandiya NATO',
            '2023-H2': 'Wagner İsyanı',
            '2025-H1': 'Trump-Zelenski',
            '2025-H2': 'Alaska Zirvesi',
            '2025-H2': 'Trump Barış Planı'
        }
        
        # X ekseni etiketlerini vurgula
        xticklabels = ax2.get_xticklabels()
        for i, label in enumerate(xticklabels):
            half_year_str = label.get_text()
            if half_year_str in olay_cizelgesi:
                label.set_color('red')
                label.set_fontweight('bold')
        
        ax2.set_title('Yarım Yıllık Konu Dağılımı - Önemli Olaylar İşaretlendi', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Yarım Yıl (H1: Ocak-Haziran, H2: Temmuz-Aralık)')
        ax2.set_ylabel('Konuşma Sayısı')
        ax2.legend(title='Konular', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()
        
        # KONU BAZINDA YARIM YILLIK DAĞILIM TABLOSU
        print("\n📋 KONU BAZINDA YARIM YILLIK DAĞILIM TABLOSU:")
        print("="*80)
        
        # Yarım yıllık konu dağılımını tablo olarak hazırla
        half_year_table = self.df.groupby(['half_year', 'topic_label']).size().unstack(fill_value=0)
        
        # Yarıyılları sırala
        half_year_table = half_year_table.sort_index()
        
        # Toplam satırı ekle
        half_year_table.loc['TOPLAM'] = half_year_table.sum()
        
        # Yüzdelik dağılımı hesapla
        half_year_percentage = half_year_table.div(half_year_table.sum(axis=1), axis=0) * 100
        
        # Tabloyu göster
        print(f"\n{'Yarım Yıl':<12} | ", end="")
        for topic_label in half_year_table.columns:
            short_label = topic_label[:20] + ('...' if len(topic_label) > 20 else '')
            print(f"{short_label:<20} | ", end="")
        print("Toplam")
        print("-" * (12 + len(half_year_table.columns) * 24))
        
        for half_year in half_year_table.index:
            if half_year == 'TOPLAM':
                print("\n" + "=" * (12 + len(half_year_table.columns) * 24))
            
            print(f"{half_year:<12} | ", end="")
            total = 0
            for topic_label in half_year_table.columns:
                count = half_year_table.loc[half_year, topic_label]
                percentage = half_year_percentage.loc[half_year, topic_label]
                total += count
                
                if count > 0:
                    print(f"{count:3d} (%{percentage:5.1f}){' ':<10}", end="")
                else:
                    print(f"{' - ':<20}", end="")
            print(f"| {total:4d}")
        
        # Konu bazında özet tablo
        print(f"\n\n📊 KONU BAZINDA ÖZET DAĞILIM:")
        print("="*80)
        
        topic_summary = []
        for topic in self.topics:
            topic_id = topic['id']
            topic_label = topic['label']
            
            # Bu konunun yarıyıllık dağılımı
            topic_half_year_dist = self.df[self.df['dominant_topic'] == topic_id].groupby('half_year').size()
            
            # Toplam belge sayısı
            total_docs = topic_half_year_dist.sum()
            
            # En aktif yarıyıl
            if len(topic_half_year_dist) > 0:
                most_active_half_year = topic_half_year_dist.idxmax()
                most_active_count = topic_half_year_dist.max()
                most_active_percentage = (most_active_count / total_docs * 100) if total_docs > 0 else 0
            else:
                most_active_half_year = '-'
                most_active_count = 0
                most_active_percentage = 0
            
            topic_summary.append({
                'Konu': f"K{topic_id+1}",
                'Konu Adı': topic_label[:50] + ('...' if len(topic_label) > 50 else ''),
                'Toplam': total_docs,
                'En Aktif Yarıyıl': most_active_half_year,
                'En Aktif Sayı': most_active_count,
                'En Aktif %': f"%{most_active_percentage:.1f}"
            })
        
        # Tabloyu göster
        summary_df = pd.DataFrame(topic_summary)
        print("\n" + summary_df.to_string(index=False))
        
        # GRAFİK 3: Önemli olaylar sonrası konu yoğunluğu - TEK BAŞINA
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        # Önemli olaylar sonrası 30 günlük periyotları analiz et
        olay_periyotlari = []
        olay_etiketler = []
        
        for olay in olaylar[:6]:  # İlk 6 önemli olay
            olay_tarih = pd.to_datetime(olay['tarih'])
            baslangic = olay_tarih - pd.Timedelta(days=15)
            bitis = olay_tarih + pd.Timedelta(days=15)
            
            # Bu periyottaki konuşmaları filtrele
            periyot_konusmalar = self.df[(self.df['date'] >= baslangic) & (self.df['date'] <= bitis)]
            
            if len(periyot_konusmalar) > 0:
                konu_dagilimi = periyot_konusmalar['dominant_topic'].value_counts(normalize=True)
                
                # Her konu için yüzdeyi al
                for konu_id in range(len(self.topics)):
                    yuzde = konu_dagilimi.get(konu_id, 0) * 100
                    olay_periyotlari.append({
                        'olay': olay['etiket'],
                        'konu': self.topics[konu_id]['label'][:30],
                        'yuzde': yuzde
                    })
        
        if olay_periyotlari:
            olay_df = pd.DataFrame(olay_periyotlari)
            pivot_df = olay_df.pivot_table(index='olay', columns='konu', values='yuzde', aggfunc='mean')
            
            # Grafik
            pivot_df.plot(kind='bar', ax=ax3, alpha=0.8, figsize=(14, 6))
            ax3.set_title('Önemli Olaylar Sonrası Konu Dağılımı (30 Günlük Periyot)', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Olay')
            ax3.set_ylabel('Konu Yüzdesi (%)')
            ax3.legend(title='Konular', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            plt.show()
        
        # GRAFİK 4: Zaman çizelgesi - olayların kronolojik gösterimi
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        
        # Basit zaman çizelgesi
        ax4.set_xlim(pd.to_datetime('2022-01-01'), pd.to_datetime('2025-12-31'))
        ax4.set_ylim(0, len(olaylar) + 1)
        
        # Olayları ekle
        for i, olay in enumerate(olaylar):
            olay_tarih = pd.to_datetime(olay['tarih'])
            ax4.plot([olay_tarih, olay_tarih], [0, i+1], '--', color=olay['renk'], alpha=0.5)
            ax4.scatter(olay_tarih, i+1, color=olay['renk'], s=100, alpha=0.8)
            ax4.text(olay_tarih + pd.Timedelta(days=30), i+1, olay['etiket'], 
                   verticalalignment='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        ax4.set_yticks(range(1, len(olaylar)+1))
        ax4.set_yticklabels([f"{i+1}. {olaylar[i]['etiket']}" for i in range(len(olaylar))])
        ax4.set_title('Önemli Olayların Kronolojik Zaman Çizelgesi', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Tarih')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Ek zaman analizi istatistikleri
        print("\n📊 ZAMAN ANALİZİ İSTATİSTİKLERİ:")
        print("-" * 50)
        
        # En aktif ay
        monthly_counts = self.df.groupby('year_month_dt').size()
        if len(monthly_counts) > 0:
            most_active_month = monthly_counts.idxmax()
            most_active_count = monthly_counts.max()
            print(f"• En aktif ay: {most_active_month.strftime('%Y-%m')} ({most_active_count} konuşma)")
        
        # En aktif yarıyıl
        half_year_counts = self.df.groupby('half_year').size()
        if len(half_year_counts) > 0:
            most_active_half_year = half_year_counts.idxmax()
            most_active_half_year_count = half_year_counts.max()
            print(f"• En aktif yarıyıl: {most_active_half_year} ({most_active_half_year_count} konuşma)")
        
        # Konu trendleri (yarım yıllık)
        print("\n📈 YARIM YILLIK KONU TRENDLERİ:")
        if len(half_year_data) >= 4:
            last_4_half_years = half_year_data.index[-4:]
            for topic_label in half_year_data.columns:
                trend_data = half_year_data.loc[last_4_half_years, topic_label]
                if trend_data.sum() > 0:
                    # Trend analizi
                    if len(trend_data) >= 2:
                        first_half = trend_data.iloc[:2].mean()
                        second_half = trend_data.iloc[2:].mean()
                        if second_half > first_half * 1.3:
                            trend = "📈 GÜÇLÜ YÜKSELİŞ"
                        elif second_half > first_half * 1.1:
                            trend = "📈 YÜKSELEN"
                        elif second_half < first_half * 0.7:
                            trend = "📉 KESKİN DÜŞÜŞ"
                        elif second_half < first_half * 0.9:
                            trend = "📉 DÜŞEN"
                        else:
                            trend = "➡️  STABİL"
                        
                        short_label = topic_label[:35] + ('...' if len(topic_label) > 35 else '')
                        print(f"  {short_label:40} → {trend}")
        
        # Önemli olayların etkisi
        print(f"\n🎯 ÖNEMLİ OLAYLARIN ANALİZİ:")
        print("-" * 50)
        for olay in olaylar[:3]:  # İlk 3 önemli olayı analiz et
            olay_tarih = pd.to_datetime(olay['tarih'])
            # Önceki 30 gün
            onceki_periyot = self.df[(self.df['date'] >= olay_tarih - pd.Timedelta(days=30)) & 
                                   (self.df['date'] < olay_tarih)]
            # Sonraki 30 gün
            sonraki_periyot = self.df[(self.df['date'] > olay_tarih) & 
                                    (self.df['date'] <= olay_tarih + pd.Timedelta(days=30))]
            
            if len(onceki_periyot) > 0 and len(sonraki_periyot) > 0:
                print(f"\n📅 {olay['etiket']} ({olay_tarih.strftime('%d.%m.%Y')}):")
                print(f"   • Önceki 30 gün: {len(onceki_periyot)} konuşma")
                print(f"   • Sonraki 30 gün: {len(sonraki_periyot)} konuşma")
                print(f"   • Değişim: {((len(sonraki_periyot)-len(onceki_periyot))/len(onceki_periyot)*100):+.1f}%")
                
                # En çok değişen konu
                onceki_konular = onceki_periyot['dominant_topic'].value_counts(normalize=True)
                sonraki_konular = sonraki_periyot['dominant_topic'].value_counts(normalize=True)
                
                for konu_id in range(len(self.topics)):
                    onceki_yuzde = onceki_konular.get(konu_id, 0) * 100
                    sonraki_yuzde = sonraki_konular.get(konu_id, 0) * 100
                    if abs(sonraki_yuzde - onceki_yuzde) > 10:  # %10'dan fazla değişim
                        konu_adi = self.topics[konu_id]['label'][:30]
                        print(f"   • {konu_adi}: {onceki_yuzde:.1f}% → {sonraki_yuzde:.1f}% "
                              f"({sonraki_yuzde-onceki_yuzde:+.1f}%)")
        
        return monthly_data
    
    def konu_bazinda_guven_grafikleri(self):
        """KONU BAZINDA DAĞILIM için güven aralıkları grafikleri"""
        print("\n📊 KONU BAZINDA GÜVEN ARALIKLARI GRAFİKLERİ OLUŞTURULUYOR...")
        
        if not hasattr(self, 'topic_confidence_stats') or not self.topic_confidence_stats:
            print("⚠️  Güven aralıkları hesaplanmamış, önce LDA analizi yapın")
            return
        
        # GRAFİK 1: Konu bazında güven aralıkları (error bar) - TEK BAŞINA
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        
        stats_df = pd.DataFrame(self.topic_confidence_stats)
        
        # Konu etiketlerini kısalt
        short_labels = []
        for label in stats_df['topic_label']:
            if len(label) > 40:
                short_labels.append(f"K{int(stats_df.loc[stats_df['topic_label']==label, 'topic_id'].iloc[0])+1}: {label[:37]}...")
            else:
                short_labels.append(f"K{int(stats_df.loc[stats_df['topic_label']==label, 'topic_id'].iloc[0])+1}: {label}")
        
        # Renkleri güven seviyesine göre belirle
        colors = []
        for level in stats_df['confidence_level']:
            if level == "ÇOK YÜKSEK":
                colors.append('green')
            elif level == "YÜKSEK":
                colors.append('limegreen')
            elif level == "ORTA":
                colors.append('orange')
            else:
                colors.append('red')
        
        # X pozisyonları
        x_pos = np.arange(len(stats_df))
        
        # Bar grafiği
        bars = ax1.bar(x_pos, stats_df['mean_confidence'], 
                      yerr=stats_df['margin_of_error'],
                      capsize=10, alpha=0.7, color=colors,
                      edgecolor='black', linewidth=1.5)
        
        # Güven aralıklarını çiz
        for i, (_, row) in enumerate(stats_df.iterrows()):
            ax1.plot([i, i], [row['ci_lower'], row['ci_upper']], 
                    color='black', linewidth=2, alpha=0.7)
            # Ortalama noktası
            ax1.scatter(i, row['mean_confidence'], color='white', 
                       s=100, zorder=5, edgecolor='black', linewidth=1.5)
            # Güven aralığı değerleri
            ax1.text(i, row['ci_upper'] + 0.01, f"{row['ci_upper']:.3f}", 
                    ha='center', fontsize=8, fontweight='bold')
            ax1.text(i, row['ci_lower'] - 0.015, f"{row['ci_lower']:.3f}", 
                    ha='center', fontsize=8, fontweight='bold')
        
        ax1.set_xlabel('Konular')
        ax1.set_ylabel('Güven Skoru (Ortalama ± %95 GA)')
        ax1.set_title(f'Konu Bazında Güven Aralıkları - %95 Güven Seviyesi (Seed: {self.random_seed})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Renk açıklaması
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='ÇOK YÜKSEK (≥0.8)'),
            Patch(facecolor='limegreen', alpha=0.7, label='YÜKSEK (0.6-0.8)'),
            Patch(facecolor='orange', alpha=0.7, label='ORTA (0.4-0.6)'),
            Patch(facecolor='red', alpha=0.7, label='DÜŞÜK (<0.4)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # GRAFİK 2: Konu dağılımı ve güven ilişkisi (scatter plot) - TEK BAŞINA
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        
        # Bubble chart: x=belge sayısı, y=ortalama güven, boyut=belge sayısı, renk=güven seviyesi
        sizes = stats_df['n_documents'] / stats_df['n_documents'].max() * 1000
        
        scatter = ax2.scatter(stats_df['n_documents'], stats_df['mean_confidence'],
                            s=sizes, c=range(len(stats_df)), 
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
        
        # Her konu için etiket
        for i, (_, row) in enumerate(stats_df.iterrows()):
            ax2.annotate(f"K{int(row['topic_id'])+1}", 
                        (row['n_documents'], row['mean_confidence']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Güven aralığı çizgileri
            ax2.plot([row['n_documents'], row['n_documents']], 
                    [row['ci_lower'], row['ci_upper']], 
                    color='gray', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Belge Sayısı (Konu Popülaritesi)')
        ax2.set_ylabel('Ortalama Güven Skoru')
        ax2.set_title('Konu Popülaritesi vs. Güven İlişkisi (Bubble Chart)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Renk bar'ını ekle
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Konu Sırası', rotation=270, labelpad=15)
        
        # Trend çizgisi
        if len(stats_df) > 1:
            z = np.polyfit(stats_df['n_documents'], stats_df['mean_confidence'], 1)
            p = np.poly1d(z)
            ax2.plot(stats_df['n_documents'], p(stats_df['n_documents']), 
                    "r--", alpha=0.5, label='Trend Çizgisi')
            ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        # GRAFİK 3: Konu güven dağılımı (violin plot) - TEK BAŞINA
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        # Her konu için güven skorlarını topla
        confidence_data = []
        konu_labels_violin = []
        
        for topic in self.topics:
            topic_id = topic['id']
            topic_confidences = self.df[self.df['dominant_topic'] == topic_id]['topic_confidence'].values
            
            if len(topic_confidences) > 0:
                confidence_data.append(topic_confidences)
                konu_labels_violin.append(f"K{topic_id+1}")
        
        # Violin plot
        violin_parts = ax3.violinplot(confidence_data, showmeans=True, showmedians=True)
        
        # Violin renklerini ayarla
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(plt.cm.tab20(i % 20))
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Mean ve median çizgilerini renklendir
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_color('blue')
        violin_parts['cmedians'].set_linewidth(2)
        
        ax3.set_xlabel('Konular')
        ax3.set_ylabel('Güven Skoru Dağılımı')
        ax3.set_title('Konu Bazında Güven Skoru Dağılımı (Violin Plot)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xticks(np.arange(1, len(konu_labels_violin) + 1))
        ax3.set_xticklabels(konu_labels_violin)
        ax3.set_ylim(0, 1.0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Açıklama
        from matplotlib.lines import Line2D
        legend_elements_violin = [
            Line2D([0], [0], color='red', linewidth=2, label='Ortalama'),
            Line2D([0], [0], color='blue', linewidth=2, label='Medyan')
        ]
        ax3.legend(handles=legend_elements_violin, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # GRAFİK 4: Konu güven ısı haritası (heatmap) - TEK BAŞINA
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        # Konu güven matrisi oluştur
        n_topics = len(self.topics)
        confidence_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(n_topics):
                if i == j:
                    # Köşegen: ortalama güven
                    conf_data = self.df[self.df['dominant_topic'] == i]['topic_confidence']
                    confidence_matrix[i, j] = conf_data.mean() if len(conf_data) > 0 else 0
                else:
                    # Köşegen dışı: konular arası ilişki (korelasyon benzeri)
                    # İki konunun birlikte görülme sıklığı
                    doc_count_i = (self.df['dominant_topic'] == i).sum()
                    doc_count_j = (self.df['dominant_topic'] == j).sum()
                    if doc_count_i > 0 and doc_count_j > 0:
                        # Jaccard benzerliği
                        intersection = ((self.df['dominant_topic'] == i) & (self.df['dominant_topic'] == j)).sum()
                        union = doc_count_i + doc_count_j - intersection
                        if union > 0:
                            confidence_matrix[i, j] = intersection / union
        
        # Heatmap
        im = ax4.imshow(confidence_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Konu etiketleri
        topic_labels_short = [f"K{i+1}" for i in range(n_topics)]
        
        # Hücre değerlerini ekle
        for i in range(n_topics):
            for j in range(n_topics):
                text = ax4.text(j, i, f"{confidence_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="black" if confidence_matrix[i, j] < 0.5 else "white",
                              fontsize=9, fontweight='bold')
        
        ax4.set_title('Konu Güven Matrisi - Isı Haritası', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Konular')
        ax4.set_ylabel('Konular')
        ax4.set_xticks(np.arange(n_topics))
        ax4.set_yticks(np.arange(n_topics))
        ax4.set_xticklabels(topic_labels_short)
        ax4.set_yticklabels(topic_labels_short)
        
        # Renk bar'ını ekle
        cbar = ax4.figure.colorbar(im, ax=ax4)
        cbar.ax.set_ylabel('Güven / Benzerlik Değeri', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()
        
        # İstatistiksel özet
        print("\n📈 KONU GÜVEN İSTATİSTİKLERİ:")
        print("-" * 50)
        
        overall_mean = stats_df['mean_confidence'].mean()
        overall_std = stats_df['mean_confidence'].std()
        
        print(f"• Genel ortalama güven: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"• En yüksek güven: Konu {int(stats_df.loc[stats_df['mean_confidence'].idxmax(), 'topic_id'])+1} "
              f"({stats_df['mean_confidence'].max():.3f})")
        print(f"• En düşük güven: Konu {int(stats_df.loc[stats_df['mean_confidence'].idxmin(), 'topic_id'])+1} "
              f"({stats_df['mean_confidence'].min():.3f})")
        print(f"• Güven aralığı genişliği: {stats_df['mean_confidence'].max() - stats_df['mean_confidence'].min():.3f}")
        
        print(f"\n📊 GÜVEN SEVİYELERİ DAĞILIMI:")
        for level in ["ÇOK YÜKSEK", "YÜKSEK", "ORTA", "DÜŞÜK"]:
            count = (stats_df['confidence_level'] == level).sum()
            if count > 0:
                percentage = (count / len(stats_df)) * 100
                print(f"  • {level}: {count} konu (%{percentage:.1f})")
        
        return stats_df
    
    def print_istatistikler_gelistirilmis(self):
        """Geliştirilmiş istatistikler"""
        print(f"\n📊 GELİŞTİRİLMİŞ İSTATİSTİKLER (Seed: {self.random_seed}):")
        print("="*70)
        
        avg_confidence = self.df['topic_confidence'].mean() * 100
        
        print(f"\n📈 GENEL PERFORMANS:")
        print(f"  • Ortalama güven: %{avg_confidence:.1f}")
        print(f"  • Toplam konuşma: {len(self.df)}")
        print(f"  • Konu sayısı: {len(self.topics)}")
        
        # Benzersiz etiket kontrolü
        unique_labels = set()
        duplicate_labels = []
        
        for topic in self.topics:
            if topic['label'] in unique_labels:
                duplicate_labels.append(topic['label'])
            unique_labels.add(topic['label'])
        
        print(f"\n🎯 KONU ÇEŞİTLİLİĞİ:")
        print(f"  • Benzersiz etiket: {len(unique_labels)}/{len(self.topics)}")
        if len(unique_labels) == len(self.topics):
            print("  ✅ MÜKEMMEL: Tüm konular benzersiz etiketlere sahip!")
        else:
            print(f"  ⚠️  UYARI: {len(duplicate_labels)} konu aynı etiketi paylaşıyor")
            for dup in set(duplicate_labels):
                print(f"     - '{dup}'")
        
        # Konu bazlı istatistikler
        print(f"\n📋 KONU BAZINDA DAĞILIM:")
        print("-" * 70)
        
        topic_stats = []
        for topic in self.topics:
            doc_count = (self.df['dominant_topic'] == topic['id']).sum()
            if doc_count > 0:
                topic_docs = self.df[self.df['dominant_topic'] == topic['id']]
                avg_conf = topic_docs['topic_confidence'].mean() * 100
                percentage = (doc_count / len(self.df)) * 100
                
                topic_stats.append({
                    'Konu': topic['id'] + 1,
                    'Etiket': topic['label'][:40] + ('...' if len(topic['label']) > 40 else ''),
                    'Doküman': doc_count,
                    '%': f"{percentage:.1f}",
                    'Ort. Güven': f"{avg_conf:.1f}%",
                    'Anahtar Kelimeler': ', '.join(topic['top_keywords'])
                })
        
        # Tablo olarak göster
        stats_df = pd.DataFrame(topic_stats)
        if not stats_df.empty:
            print("\n" + stats_df.to_string(index=False))
        
        # Güven aralıkları istatistikleri
        if hasattr(self, 'topic_confidence_stats') and self.topic_confidence_stats:
            print(f"\n📊 KONU GÜVEN ARALIKLARI ÖZETİ:")
            print("-" * 50)
            
            conf_stats_df = pd.DataFrame(self.topic_confidence_stats)
            
            for _, row in conf_stats_df.iterrows():
                print(f"\n• Konu {int(row['topic_id'])+1}: {row['topic_label'][:40]}...")
                print(f"  → Belge sayısı: {row['n_documents']}")
                print(f"  → Ortalama güven: {row['mean_confidence']:.3f}")
                print(f"  → %95 Güven Aralığı: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
                print(f"  → Hata payı: ±{row['margin_of_error']:.3f}")
                print(f"  → Seviye: {row['confidence_level']} {row['level_color']}")
    
    def run_gelistirilmis_analiz(self):
        """Geliştirilmiş analizi çalıştır"""
        print("\n" + "="*70)
        print("🚀 GELİŞTİRİLMİŞ LDA ANALİZİ BAŞLATILIYOR")
        print("="*70)
        
        try:
            # Konu sayısı
            if self.n_topics is None:
                n_topics = 5  # Varsayılan
            else:
                n_topics = self.n_topics
            
            print(f"✅ Konu sayısı: {n_topics}")
            
            # DTM oluştur
            self.create_dtm_gelistirilmis(n_topics=n_topics)
            
            # LDA eğitimi
            self.perform_lda_gelistirilmis(n_topics=n_topics)
            
            # Zaman analizi grafikleri (ALT ALTA)
            self.zaman_analizi_grafikleri()
            
            # Konu bazında güven aralıkları grafikleri
            self.konu_bazinda_guven_grafikleri()
            
            # İstatistikler
            self.print_istatistikler_gelistirilmis()
            
            # Sonuçları kaydet
            self.save_results_gelistirilmis()
            
            print("\n" + "="*70)
            print(f"✅ GELİŞTİRİLMİŞ ANALİZ BAŞARIYLA TAMAMLANDI!")
            print("="*70)
            
            return {
                'success': True,
                'topics': self.topics,
                'avg_confidence': self.df['topic_confidence'].mean(),
                'random_seed': self.random_seed,
                'n_topics': n_topics
            }
            
        except Exception as e:
            print(f"\n❌ HATA: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def save_results_gelistirilmis(self):
        """Sonuçları kaydet"""
        output_dir = f'gelistirilmis_lda_results_seed_{self.random_seed}'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 SONUÇLAR '{output_dir}' KLASÖRÜNE KAYDEDİLİYOR...")
        
        # Tüm sonuçları kaydet
        self.df.to_csv(f'{output_dir}/tum_sonuclar.csv', index=False, encoding='utf-8-sig')
        print(f"✓ Tüm sonuçlar kaydedildi: {output_dir}/tum_sonuclar.csv")
        
        # Konu özeti
        summary_data = []
        for topic in self.topics:
            doc_count = (self.df['dominant_topic'] == topic['id']).sum()
            topic_docs = self.df[self.df['dominant_topic'] == topic['id']]
            
            summary_data.append({
                'konu_no': topic['id'] + 1,
                'konu_etiketi': topic['label'],
                'dokuman_sayisi': doc_count,
                'yuzde': (doc_count / len(self.df)) * 100,
                'ortalama_guven': topic_docs['topic_confidence'].mean() * 100,
                'anahtar_kelimeler': ', '.join(topic['keywords'][:10])
            })
        
        pd.DataFrame(summary_data).to_csv(
            f'{output_dir}/konu_ozeti.csv', 
            index=False, encoding='utf-8-sig'
        )
        print(f"✓ Konu özeti kaydedildi: {output_dir}/konu_ozeti.csv")
        
        # Güven aralıkları
        if hasattr(self, 'topic_confidence_stats') and self.topic_confidence_stats:
            conf_df = pd.DataFrame(self.topic_confidence_stats)
            conf_df.to_csv(
                f'{output_dir}/guven_araliklari.csv',
                index=False, encoding='utf-8-sig'
            )
            print(f"✓ Güven aralıkları kaydedildi: {output_dir}/guven_araliklari.csv")
        
        print(f"\n📁 TÜM SONUÇLAR: {os.path.abspath(output_dir)}/")


# ============================================================================
# ANA PROGRAM
# ============================================================================

def main_gelistirilmis():
    """Ana program"""
    
    CSV_PATH = "smart_stopwords_results/filtered_speeches.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Dosya bulunamadı: {CSV_PATH}")
        return
    
    print("\n🎯 GELİŞTİRİLMİŞ PUTİN KONUŞMALARI ANALİZİ")
    print("="*70)
    
    # Random seed
    seed_input = input("Random seed girin (boş=42): ").strip()
    RANDOM_SEED = int(seed_input) if seed_input else 42
    
    # Konu sayısı
    topics_input = input("Konu sayısı girin (2-8, boş=5): ").strip()
    if topics_input:
        N_TOPICS = int(topics_input)
        if N_TOPICS < 2 or N_TOPICS > 8:
            print("⚠️  2-8 arası olmalı, varsayılan 5 kullanılıyor")
            N_TOPICS = 5
    else:
        N_TOPICS = 5
    
    print(f"\n✅ AYARLAR:")
    print(f"  • Random seed: {RANDOM_SEED}")
    print(f"  • Konu sayısı: {N_TOPICS}")
    print(f"  • Zaman analizi: AKTİF")
    print(f"  • Önemli olay işaretleyicileri: AKTİF (9 önemli olay)")
    print(f"  • Konu güven aralıkları: AKTİF (4 yeni grafik)")
    print(f"  • Güncellenmiş stopwords: AKTİF")
    
    input(f"\n⏎ Analizi başlatmak için ENTER'a basın (Seed: {RANDOM_SEED})...")
    
    analyzer = PutinLDAGelistirilmis(CSV_PATH, random_seed=RANDOM_SEED, n_topics=N_TOPICS)
    results = analyzer.run_gelistirilmis_analiz()
    
    if results.get('success', False):
        print(f"\n✨ ANALİZ TAMAMLANDI!")
        print(f"  • Konu sayısı: {results['n_topics']}")
        print(f"  • Ortalama güven: %{results['avg_confidence']*100:.1f}")
        print(f"  • Zaman analizi grafikleri: 4 GRAFİK OLUŞTURULDU")
        print(f"  • Konu güven grafikleri: 4 GRAFİK OLUŞTURULDU")
        print(f"  • Önemli olaylar: 9 olay işaretlendi")
        
        # Benzersiz etiket kontrolü
        unique_labels = set(t['label'] for t in results['topics'])
        if len(unique_labels) == len(results['topics']):
            print(f"  ✅ KONU ETİKETLERİ: {len(unique_labels)}/{len(results['topics'])} benzersiz")
        else:
            print(f"  ⚠️  KONU ETİKETLERİ: {len(unique_labels)}/{len(results['topics'])} benzersiz")
        
        print(f"\n📁 SONUÇLAR: 'gelistirilmis_lda_results_seed_{RANDOM_SEED}/' klasöründe")
    
    else:
        print(f"\n❌ HATA: {results.get('error', 'Bilinmeyen')}")


if __name__ == "__main__":
    main_gelistirilmis()


