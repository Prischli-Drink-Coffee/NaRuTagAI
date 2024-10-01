import numpy as np 
import pandas as pd
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

@dataclass
class TagEmbeddingGeneration:
    taxonomy_path: str = 'src/baseline/IAB_tags.csv'
    pretrained_model: str = 'DeepPavlov/rubert-base-cased-sentence'

    def __post_init__(self):
        self.taxonomy = pd.read_csv(self.taxonomy_path)
        self.model = SentenceTransformer(self.pretrained_model)

    def get_reference_tags(self):
        tags = {}
        for _, row in self.taxonomy.iterrows():
            if isinstance(row['Уровень 1 (iab)'], str):
                tags[row['Уровень 1 (iab)']] = self.model.encode(row['Уровень 1 (iab)'], verbose=False, convert_to_tensor=True).cpu().numpy()
            if isinstance(row['Уровень 2 (iab)'], str):
                tags[row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)']] = self.model.encode(row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)'], verbose=False, convert_to_tensor=True).cpu().numpy()
            if isinstance(row['Уровень 3 (iab)'], str):
                tags[row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)']+": "+row['Уровень 3 (iab)']] = self.model.encode(row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)']+": "+row['Уровень 3 (iab)'], verbose=False, convert_to_tensor=True).cpu().numpy()
        return tags
    
    def get_reference_embedding(self):
        tags = self.get_reference_tags()
        tags_list = list(self.tags.keys())
        vectors = np.array(list(self.tags.values()))
        return tags, tags_list, vectors
    
    def get_predicted_embedding(self, prediction):
        return self.model.encode(' '.join(prediction), verbose=False)





