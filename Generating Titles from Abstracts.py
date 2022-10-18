#https://www.kaggle.com/code/vedanth2000/transformers-generating-titles-from-abstracts
#https://www.kaggle.com/code/mathurinache/simplet5-generating-one-line-summary-of-papers
#https://www.kaggle.com/code/maartengr/topic-modeling-arxiv-abstract-with-bertopic

data_file = r"C:\Users\micha\Tensorflow_datasets\arxiv-metadata-oai-snapshot.json"
#https://www.kaggle.com/datasets/1b6883fb66c5e7f67c697c2547022cc04c9ee98c3742f9a4d6c671b4f4eda591/code?resource=download

# pip install bertopic
# pip install simplet5


import json, re
import pandas as pd
from tqdm import tqdm


# https://arxiv.org/help/api/user-manual
category_map = {'astro-ph': 'Astrophysics',
'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP': 'Earth and Planetary Astrophysics',
'astro-ph.GA': 'Astrophysics of Galaxies',
'astro-ph.HE': 'High Energy Astrophysical Phenomena',
'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
'astro-ph.SR': 'Solar and Stellar Astrophysics',
'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci': 'Materials Science',
'cond-mat.other': 'Other Condensed Matter',
'cond-mat.quant-gas': 'Quantum Gases',
'cond-mat.soft': 'Soft Condensed Matter',
'cond-mat.stat-mech': 'Statistical Mechanics',
'cond-mat.str-el': 'Strongly Correlated Electrons',
'cond-mat.supr-con': 'Superconductivity',
'cs.AI': 'Artificial Intelligence',
'cs.AR': 'Hardware Architecture',
'cs.CC': 'Computational Complexity',
'cs.CE': 'Computational Engineering, Finance, and Science',
'cs.CG': 'Computational Geometry',
'cs.CL': 'Computation and Language',
'cs.CR': 'Cryptography and Security',
'cs.CV': 'Computer Vision and Pattern Recognition',
'cs.CY': 'Computers and Society',
'cs.DB': 'Databases',
'cs.DC': 'Distributed, Parallel, and Cluster Computing',
'cs.DL': 'Digital Libraries',
'cs.DM': 'Discrete Mathematics',
'cs.DS': 'Data Structures and Algorithms',
'cs.ET': 'Emerging Technologies',
'cs.FL': 'Formal Languages and Automata Theory',
'cs.GL': 'General Literature',
'cs.GR': 'Graphics',
'cs.GT': 'Computer Science and Game Theory',
'cs.HC': 'Human-Computer Interaction',
'cs.IR': 'Information Retrieval',
'cs.IT': 'Information Theory',
'cs.LG': 'Machine Learning',
'cs.LO': 'Logic in Computer Science',
'cs.MA': 'Multiagent Systems',
'cs.MM': 'Multimedia',
'cs.MS': 'Mathematical Software',
'cs.NA': 'Numerical Analysis',
'cs.NE': 'Neural and Evolutionary Computing',
'cs.NI': 'Networking and Internet Architecture',
'cs.OH': 'Other Computer Science',
'cs.OS': 'Operating Systems',
'cs.PF': 'Performance',
'cs.PL': 'Programming Languages',
'cs.RO': 'Robotics',
'cs.SC': 'Symbolic Computation',
'cs.SD': 'Sound',
'cs.SE': 'Software Engineering',
'cs.SI': 'Social and Information Networks',
'cs.SY': 'Systems and Control',
'econ.EM': 'Econometrics',
'eess.AS': 'Audio and Speech Processing',
'eess.IV': 'Image and Video Processing',
'eess.SP': 'Signal Processing',
'gr-qc': 'General Relativity and Quantum Cosmology',
'hep-ex': 'High Energy Physics - Experiment',
'hep-lat': 'High Energy Physics - Lattice',
'hep-ph': 'High Energy Physics - Phenomenology',
'hep-th': 'High Energy Physics - Theory',
'math.AC': 'Commutative Algebra',
'math.AG': 'Algebraic Geometry',
'math.AP': 'Analysis of PDEs',
'math.AT': 'Algebraic Topology',
'math.CA': 'Classical Analysis and ODEs',
'math.CO': 'Combinatorics',
'math.CT': 'Category Theory',
'math.CV': 'Complex Variables',
'math.DG': 'Differential Geometry',
'math.DS': 'Dynamical Systems',
'math.FA': 'Functional Analysis',
'math.GM': 'General Mathematics',
'math.GN': 'General Topology',
'math.GR': 'Group Theory',
'math.GT': 'Geometric Topology',
'math.HO': 'History and Overview',
'math.IT': 'Information Theory',
'math.KT': 'K-Theory and Homology',
'math.LO': 'Logic',
'math.MG': 'Metric Geometry',
'math.MP': 'Mathematical Physics',
'math.NA': 'Numerical Analysis',
'math.NT': 'Number Theory',
'math.OA': 'Operator Algebras',
'math.OC': 'Optimization and Control',
'math.PR': 'Probability',
'math.QA': 'Quantum Algebra',
'math.RA': 'Rings and Algebras',
'math.RT': 'Representation Theory',
'math.SG': 'Symplectic Geometry',
'math.SP': 'Spectral Theory',
'math.ST': 'Statistics Theory',
'math-ph': 'Mathematical Physics',
'nlin.AO': 'Adaptation and Self-Organizing Systems',
'nlin.CD': 'Chaotic Dynamics',
'nlin.CG': 'Cellular Automata and Lattice Gases',
'nlin.PS': 'Pattern Formation and Solitons',
'nlin.SI': 'Exactly Solvable and Integrable Systems',
'nucl-ex': 'Nuclear Experiment',
'nucl-th': 'Nuclear Theory',
'physics.acc-ph': 'Accelerator Physics',
'physics.ao-ph': 'Atmospheric and Oceanic Physics',
'physics.app-ph': 'Applied Physics',
'physics.atm-clus': 'Atomic and Molecular Clusters',
'physics.atom-ph': 'Atomic Physics',
'physics.bio-ph': 'Biological Physics',
'physics.chem-ph': 'Chemical Physics',
'physics.class-ph': 'Classical Physics',
'physics.comp-ph': 'Computational Physics',
'physics.data-an': 'Data Analysis, Statistics and Probability',
'physics.ed-ph': 'Physics Education',
'physics.flu-dyn': 'Fluid Dynamics',
'physics.gen-ph': 'General Physics',
'physics.geo-ph': 'Geophysics',
'physics.hist-ph': 'History and Philosophy of Physics',
'physics.ins-det': 'Instrumentation and Detectors',
'physics.med-ph': 'Medical Physics',
'physics.optics': 'Optics',
'physics.plasm-ph': 'Plasma Physics',
'physics.pop-ph': 'Popular Physics',
'physics.soc-ph': 'Physics and Society',
'physics.space-ph': 'Space Physics',
'q-bio.BM': 'Biomolecules',
'q-bio.CB': 'Cell Behavior',
'q-bio.GN': 'Genomics',
'q-bio.MN': 'Molecular Networks',
'q-bio.NC': 'Neurons and Cognition',
'q-bio.OT': 'Other Quantitative Biology',
'q-bio.PE': 'Populations and Evolution',
'q-bio.QM': 'Quantitative Methods',
'q-bio.SC': 'Subcellular Processes',
'q-bio.TO': 'Tissues and Organs',
'q-fin.CP': 'Computational Finance',
'q-fin.EC': 'Economics',
'q-fin.GN': 'General Finance',
'q-fin.MF': 'Mathematical Finance',
'q-fin.PM': 'Portfolio Management',
'q-fin.PR': 'Pricing of Securities',
'q-fin.RM': 'Risk Management',
'q-fin.ST': 'Statistical Finance',
'q-fin.TR': 'Trading and Market Microstructure',
'quant-ph': 'Quantum Physics',
'stat.AP': 'Applications',
'stat.CO': 'Computation',
'stat.ME': 'Methodology',
'stat.ML': 'Machine Learning',
'stat.OT': 'Other Statistics',
'stat.TH': 'Statistics Theory'}



def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line
            
titles = []
abstracts = []
years = []
categories = []
ids = []
authors_parsed = []
authors = []
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('journal-ref')
    try:
        year = int(ref[-4:]) 
        if 2019 < year <= 2021:
            categories.append(category_map[paper_dict.get('categories').split(" ")[0]])
            years.append(year)
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
            ids.append(paper_dict.get('id'))
            authors_parsed.append(paper_dict.get('authors_parsed'))
            authors.append(paper_dict.get('authors'))

    except:
        pass 

print(len(titles), len(abstracts), len(years), len(categories))

papers = pd.DataFrame({
    'id' : ids,
    'title': titles,
    'authors': authors,
    'authors parsed': authors_parsed,
    'abstract': abstracts,
    'year': years
})

papers = papers.dropna()
papers["title"] = papers["title"].apply(lambda x: re.sub('\s+',' ', x))
papers["abstract"] = papers["abstract"].apply(lambda x: re.sub('\s+',' ', x))








from simplet5 import SimpleT5

# simpleT5 expects training and validation dataframes to have 2 columns: "source_text" and "target_text"
papers = papers[['abstract','title']]
papers.columns = ["source_text", "target_text"]
# let's add a prefix to source_text, to uniquely identify kind of task we are performing on the data, in this case --> "summarize"
papers['source_text'] = "summarize: "+ papers['source_text']

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(papers, test_size=0.1)

# instatntiate
model = SimpleT5()
model.from_pretrained("t5","t5-base")

model.train(train_df=train_df, eval_df=test_df, source_max_token_len=512, target_max_token_len=128, max_epochs=5, batch_size=8 , use_gpu=False)
## simpleT5 saves your model at every epoch in "outputs" folder (default)
#model.load_model("outputs/SimpleT5-epoch-4-train-loss-1.1588", use_gpu=True)

sample_abstracts = test_df.sample(10)
for i, abstract in sample_abstracts.iterrows():
    print(f"===== Abstract =====")
    print(abstract['source_text'])
    summary= model.predict(abstract['source_text'])[0]

    print(f"\n===== Actual Title =====")
    print(f"{abstract['target_text']}")
    print(f"\n===== Generated Title =====")
    print(f"{summary}")
    print("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")



# or
'''
! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -U transformers
!pip install -U simpletransformers  
'''

papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']
papers = papers.dropna()
eval_df = papers.sample(frac=0.2, random_state=101)
train_df = papers.drop(eval_df.index)


import logging

import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df['prefix'] = "summarize"
eval_df['prefix'] = "summarize"


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create T5 Model
model = T5Model("t5-small", args=model_args, use_cuda=False)

model.train_model(train_df)

results = model.eval_model(eval_df)



random_num = 999
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')






#### then

from bertopic import BERTopic

topic_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L12-v2", min_topic_size=50)
topics, _ = topic_model.fit_transform(abstracts); len(topic_model.get_topic_info())

topic_model.get_topic_info().head(10)
topic_model.visualize_barchart(top_n_topics=9, height=700)
topic_model.visualize_term_rank()
topic_model.visualize_term_rank(log_scale=True)
topic_model.visualize_topics(top_n_topics=50)
topic_model.visualize_hierarchy(top_n_topics=50, width=800)
topic_model.visualize_heatmap(n_clusters=20, top_n_topics=100)
topics_over_time = topic_model.topics_over_time(abstracts, topics, years)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20, width=900, height=500)
topics_per_class = topic_model.topics_per_class(abstracts, topics, classes=categories)
topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10, width=900)
