import pandas as pd
from tqdm import tqdm
from google_play_scraper import Sort, reviews

# Google play scraper: https://github.com/JoMingyu/google-play-scraper
# !pip install google_play_scraper

# Aplicativos a serem analisados
apps_ids = [
    'com.itau', 
    'com.nu.production', 
    'br.com.bb.android', 
    'com.bradesco',
    'com.santander.app', 
    'com.c6bank.app', 
    'br.com.xp.carteira', 
    'br.com.intermedium'
]

# Web scraping para obter as avaliações dos aplicativos e salvar no CSV
def obter_reviews():
    app_reviews = []
    for ap in tqdm(apps_ids):
        rvs, _ = reviews(
            ap,
            lang='pt',
            country='br',
            sort=Sort.NEWEST,
            count=8000,
        )
        for r in rvs:
            r['sortOrder'] = 'newest'
            r['appId'] = ap
        app_reviews.extend(rvs)
    
    # Convertendo as avaliações para DataFrame
    df_reviews = pd.DataFrame(app_reviews)
    
    # Salvando as avaliações em um arquivo CSV
    df_reviews.to_csv('app_reviews.csv', index=False)
    return df_reviews