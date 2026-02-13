# Mapping Strategic Narratives: Optimized Topic Modelling and Temporal Evolution in Putin’s Ukraine Discourse (2022-2025) 📊

This study examines the dominant themes and strategic framing in Russian President Vladimir Putin’s discourse on Ukraine within the context of the geopolitical crisis and domestic political consolidation between February 2022 and December 2025. From more than 2,000 speeches published on the Kremlin’s official website, a corpus of 126 presidential speeches directly related to Ukraine, comprising approximately 750,000 words, was analysed. 

1. **Smart Stopwords Detector:** Identifies and filters out meaningless, repetitive, or contextually insignificant words from speech texts using statistical and ensemble methods.
2. **Enhanced Topic Modeling (LDA):** Extracts main themes from cleaned texts using the Latent Dirichlet Allocation (LDA) algorithm, analyzes their evolution over time, and presents them with confidence intervals.

## 🚀 Features

- **Data Collection and Preprocessing:** Text extraction from `.docx` files, saving in CSV/Excel format.
- **Advanced Stopwords Detection:**
  - Multiple statistical methods (High Frequency, High Document Frequency, Low TF-IDF, Short Words)
  - Ensemble voting system to identify the most meaningful stopwords
  - Optimized stopwords list based on target text reduction percentage (35% default)
  - Addition of context-specific words for Putin's speeches
- **Enhanced LDA Topic Modeling:**
  - N-gram (2-4 word phrases) support for more meaningful topic extraction
  - Automatic, interpretable labeling of topics
  - **Time Series Analysis:** Visualization of monthly and semi-annual topic distribution correlated with important historical events
  - **Statistical Confidence Intervals:** 95% confidence interval calculation for the reliability of dominant topic assignments
- **Visualization:**
  - Stacked area charts showing topic evolution over time
  - Bubble charts showing relationship between topic popularity and confidence scores
  - Violin plots showing topic-based confidence score distributions
  - Heat maps showing inter-topic similarity matrices
- **Save Results:** All analysis results are automatically saved to CSV files

## 🛠️ Technologies Used

- Python 3.x
- **Data Processing:** Pandas, NumPy
- **Natural Language Processing (NLP):** scikit-learn (CountVectorizer, TfidfVectorizer, LatentDirichletAllocation)
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy
- **Other:** `python-docx`, `openpyxl`, `collections`, `re`, `os`

## 📁 Project Structure

```
putin-speeches-analysis/
├── speech.docx                          # Raw Word file containing speech data
├── speech.csv                            # Raw CSV created in the first stage
├── main.py                              # Main execution script
├── main.ipynb                                   
├── smart_stopwords_results/               # Stopwords analysis outputs
│   ├── smart_stopwords.txt
│   ├── filtered_speeches.csv
│   └── analysis_report.txt
├── enhanced_lda_results_seed_[SEED]/     # LDA analysis outputs
│   ├── all_results.csv
│   ├── topic_summary.csv
│   └── confidence_intervals.csv
└── README.md
```

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/putin-speeches-analysis.git
   cd putin-speeches-analysis
   ```

2. **Install required libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy python-docx openpyxl
   ```

## ⚙️ Usage

1. **Data Preparation:** Place your `speech.docx` file in the project root directory. Expected file format:
   ```
   #
   Speech Title
   YYYY-MM-DD
   URL
   Speaker: Keywords (optional)
   Speech text goes here...
   #
   Next Speech Title
   ...
   ```

2. **Run the Analysis:**
   ```bash
   python main.py
   ```

3. **LDA Analysis Parameters:**
   - Enter random seed (leave empty for default 42)
   - Enter number of topics (2-8, leave empty for default 5)

## 📊 Sample Outputs

### 1. Smart Stopwords List (`smart_stopwords.txt`)
```
# SMART STOPWORDS DETECTOR - OPTIMIZED RESULTS
WORD | FREQUENCY | PERCENTAGE | DOCUMENT_PERCENTAGE
--------------------------------------------------
the                  |   42,000 | % 5.62 | %100.0
and                  |   23,626 | % 3.16 | % 99.2
to                   |   21,506 | % 2.88 | %100.0
of                   |   20,801 | % 2.78 | %100.0
...
```

### 2. Topic Summary (`topic_summary.csv`)
| topic_no | topic_label | document_count | percentage | avg_confidence | key_keywords |
|----------|-------------|----------------|------------|----------------|--------------|
|1|STRATEGIC DEFENSE: NUCLEAR WEAPON SYSTEMS|49|38.9|81.5|"soviet union, artificial intelligence, south africa, russia china, kiev regime, central bank, global south, economic growth, bilateral relations, ukrainian side"|
|2|UKRAINE: KIEV REGIME AND DONBAS INDEPENDENCE|31|24.6|83.6|"defence ministry, soviet union, military personnel, defence industry, kiev regime, donetsk lugansk, constitutional court, great patriotic, lugansk republics, donetsk lugansk republics"|
|3|HISTORICAL MEMORIES: LENINGRAD SIEGE|14|11.1|91.9|"great patriotic, kursk region, bryansk region, middle east, historical memory, great victory, energy sector, defence ministry, interior ministry, nuclear power"|
|4|ECONOMIC DEVELOPMENT: TECHNOLOGY AND INDUSTRY POLICY|5|3.9|80.9|"economic forum, central bank, global economy, economic growth, national projects, economic development, international economic, social sphere, asia pacific, high tech"|
|5|TERROR ATTACKS: CRIMEAN BRIDGE SECURITY|27|21.4|86.8|"middle east, defence ministry, kiev regime, terrorist attack, group forces, general staff, crimean bridge, belgorod region, soviet union, andrei kolesnikov"|


### 3. Visual Outputs

When the project runs, it automatically generates 4 main graphs:

- **Topic Distribution Over Time:** Stacked area chart showing monthly topic evolution
- **Topic Popularity vs. Confidence Score:** Bubble chart analyzing relationships
- **Confidence Score Distribution:** Violin plots showing topic-based distributions
- **Topic Similarity Matrix:** Heat map showing inter-topic relationships

## 📈 Analysis Metrics

- **Perplexity:** Model performance metric (lower values are better)
- **Confidence Score:** Probability assigned to the dominant topic for each speech (0-1 range)
- **Confidence Interval:** 95% confidence level for topic-based average confidence
- **Topic Popularity:** Number and percentage of speeches per topic

## 🎯 Key Events Marked on Timeline

The time analysis charts include markers for significant events:

- Ukraine Invasion (February 24, 2022)
- SWIFT Sanctions (March 2022)
- Crimean Bridge Explosion (October 2022)
- Kherson Retaken (November 2022)
- Wagner Mutiny (June 2023)
- Finland NATO Accession (April 2023)
- Trump-Zelensky Crisis (February 2025)
- Trump-Putin Alaska Summit (August 2025)
- Trump Peace Plan (November 2025)


## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or feedback: [hasan.ucuzal@inonu.edu.tr]

---

**Note:** This project is developed for academic and analytical purposes only. Findings may vary depending on the dataset used and model parameters.
