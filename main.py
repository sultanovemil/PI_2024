import os
import re
import streamlit as st
import googleapiclient.discovery
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Анализатор комментариев :red[YouTube] :sunglasses:')

# Получаем YouTube API KEY из secrets
DEVELOPER_KEY = os.getenv('API_KEY_YOUTUBE')

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


def change_url():
    st.session_state.start = False


if "start" not in st.session_state:
    st.session_state.start = False

# Получаем id видеоролика из URL для отправки запроса
url = st.text_input(label="Enter URL from YouTube", on_change=change_url)
vidID = extract_video_id(url)
if  vidID != "":
    if btn_start := st.button('Загрузить комментарии'):
        st.session_state.start = True

if st.session_state.start:
    # Запрос к YouTube API для получения комментариев к видео
    api_service_name = "youtube"
    api_version = "v3"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=vidID,
        maxResults=100
    )
    response = request.execute()
    comments = []
    # Преобразуем полученные комментарии в DataFrame
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['updatedAt'],
            comment['likeCount'],
            comment['textDisplay']
        ])
    comments_df = pd.DataFrame(comments,
                               columns=['author',
                                        'published_at',
                                        'updated_at',
                                        'like_count',
                                        'text'])

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
