import os
import re
import streamlit as st
import googleapiclient.discovery
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Анализатор комментариев :red[YouTube] :sunglasses:')


# Инициализируем модель Hugging Face для анализа тональности текста
cls_sent = pipeline("sentiment-analysis",
                    "blanchefort/rubert-base-cased-sentiment")


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube video URL.
    Args:       url (str): The YouTube video URL.
    Returns:    str: The extracted video ID,
                or an empty string if the URL is not valid.
    """
    pattern = r"(?<=v=)[\w-]+(?=&|\b)"
    match = re.search(pattern, url)
    if match:
        return match.group()
    else:
        return ""


def download_comments(video_id: str) -> pd.DataFrame:
    """
    Downloads comments from a YouTube video based on the provided video ID and returns them as a DataFrame.
    Args: video_id (str): The video ID of the YouTube video.
    Returns: DataFrame: A DataFrame containing the downloaded comments from the video.
    """
    DEV_KEY = os.getenv('API_KEY_YOUTUBE')  
    youtube = googleapiclient.discovery.build("youtube",
                                              "v3",
                                              developerKey=DEV_KEY)
    request = youtube.commentThreads().list(part="snippet",
                                            videoId=video_id,
                                            maxResults=100)
    response = request.execute()
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([comment['authorDisplayName'],
                        comment['publishedAt'],
                        comment['updatedAt'],
                        comment['likeCount'],
                        comment['textDisplay'],])
    return pd.DataFrame(comments,
                        columns=['author',
                                'published_at',
                                'updated_at',
                                'like_count',
                                'text',])


def change_url():
    st.session_state.start = False


if "start" not in st.session_state:
    st.session_state.start = False

# Получаем id видеоролика из URL для отправки запроса
url = st.text_input(label="Enter URL from YouTube", on_change=change_url)
video_id = extract_video_id(url)
if  video_id != "":
    if btn_start := st.button('Загрузить комментарии'):
        st.session_state.start = True

if st.session_state.start:    
    comments_df = download_comments(video_id)

    # Получаем таблицу с комментариями на странице
    st.header('Комментарии из YouTube')
    selected_columns = ['text', 'author', 'published_at']
    comments_df = comments_df[selected_columns]

    res_list = []
    # Анализируем тональность комментария с помощью модели Hugging Face
    with st.spinner('Идет процесс обработки данных...'):
        res_list = cls_sent(comments_df['text'].to_list())
    s_label = f'Готово! Обработано {len(res_list)} комментариев.'
    st.success(s_label)

    # Выводим таблицу с результатами на странице
    full_df = pd.concat([pd.DataFrame(res_list), comments_df], axis=1)
    st.write(full_df)
    st.markdown('***')

    # Выводим heatmap комментариев по часам и датам
    st.header('Комментарии по часам и датам')
    full_df['published_at'] = pd.to_datetime(full_df['published_at'])
    full_df['Date'] = full_df['published_at'].dt.date
    full_df['Hour'] = full_df['published_at'].dt.hour
    pivot_table = full_df.pivot_table(index='Hour',
                                      columns='Date',
                                      values='text',
                                      aggfunc='count')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu')
    plt.title('Количество комментариев по часам и датам')
    plt.xlabel('Дата')
    plt.ylabel('Час')
    st.pyplot(plt)
    st.markdown('***')

    # Создаем круговую диаграмму
    st.header('Эмоциональная окраска комментариев на YouTube')
    data = full_df['label'].value_counts()
    fig, ax = plt.subplots()
    plt.title("Эмоциональная окраска комментариев на YouTube")
    label = full_df['label'].unique()
    ax.pie(data, labels=label, autopct='%1.1f%%')
    st.pyplot(fig)
