import os
import pandas as pd
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors


def searchVideosByListOfIds(ids_list):
    api_key = os.environ.get('YT_API')

    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key)

    request = youtube.videos().list(
        part="snippet",
        maxResults=50,
        id=ids_list,
    )
    response = request.execute()
    return response

def fillCategoryIds(df, step=50):
    # 50 is max results count returned by YT API
    
    for i in range(0,df.shape[0],step):
        ids_list = df[i:i+step]["video_id"].values.tolist()

        try:
            result = searchVideosByListOfIds(ids_list)
        except:
            df.loc[i:i+step,:]["category_id"] = ['' for _ in range(step)]
            continue
            
        cats_list = [item['snippet']['categoryId'] for item in result['items']]

        # in case if any id is invalid and returned result list is too short
        if len(cats_list) < step:
            df.loc[i:i+step, :]["category_id"] = ''
            continue

        df[i:i+step]["category_id"] = cats_list
        
    return df