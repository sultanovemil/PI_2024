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
# Кэшируем ресурс для одной загрузки модели на все сессии
@st.cache_resource
def load_model():
    """
    Loads the 'blanchefort/rubert-base-cased-sentiment' model from HuggingFace
    and saves to cache for consecutive loads.
    """
    model = pipeline(
        "sentiment-analysis",
        "blanchefort/rubert-base-cased-sentiment")
    return model


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube video URL.
    """
    pattern = r"(?<=v=)[\w-]+(?=&|\b)"
    match = re.search(pattern, url)
    if match:
        return match.group()
    else:
        return ""


def download_comments(video_id: str) -> pd.DataFrame:
    """
    Downloads comments from a YouTube video based on the provided video ID
    and returns them as a DataFrame.
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
        comments.append([
                        comment['authorDisplayName'],
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


def analyze_emotions_in_comments(df: pd.DataFrame) -> tuple:
    """
    Takes a DataFrame with comments,
    processes the emotional sentiment of each comment in the DataFrame
    """
    model = load_model()
    selected_columns = ['text', 'author', 'published_at']
    df = df[selected_columns]
    res_list = []
    res_list = model(df['text'][:513].to_list())
    full_df = pd.concat([pd.DataFrame(res_list), df], axis=1)
    return (full_df, len(res_list))


def plot_heatmap_from_dataframe(df: pd.DataFrame) -> plt:
    """
    Visualizes the data from the input DataFrame
    and returns a matplotlib plot object.
    """
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['Date'] = df['published_at'].dt.date
    df['Hour'] = df['published_at'].dt.hour
    pivot_table = df.pivot_table(index='Hour',
                                columns='Date',
                                values='text',
                                aggfunc='count')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table,
                cmap='YlGnBu')
    plt.title('Количество комментариев по часам и датам')
    plt.xlabel('Дата')
    plt.ylabel('Час')
    return plt


def visualize_data(df: pd.DataFrame):
    """
    Visualizes the data from the input DataFrame
    and returns a matplotlib figure object.
    """
    data = df['label'].value_counts()
    fig, ax = plt.subplots()
    plt.title("Эмоциональная окраска комментариев на YouTube")
    label = df['label'].unique()
    ax.pie(data, labels=label, autopct='%1.1f%%')
    return fig


def change_url():
    st.session_state.start = False


if "start" not in st.session_state:
    st.session_state.start = False

# Получаем id видеоролика из URL для отправки запроса
url = st.text_input(label="Enter URL from YouTube", on_change=change_url)
video_id = extract_video_id(url)
if video_id != "":
    if btn_start := st.button('Загрузить комментарии'):
        st.session_state.start = True

if st.session_state.start:
    # Выводим таблицу с результатами на странице
    comments_df = download_comments(video_id)
    with st.spinner('Analyzing comments...'):
        full_df, num_comments = analyze_emotions_in_comments(comments_df)
        st.success(f'Готово! Обработано {num_comments} комментариев.')
    st.write(full_df)
    st.markdown('***')

    # Выводим heatmap комментариев по часам и датам
    st.pyplot(plot_heatmap_from_dataframe(full_df))
    st.markdown('***')

    # Выводим круговую диаграмму
    st.pyplot(visualize_data(full_df))
