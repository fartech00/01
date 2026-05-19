import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path
import re
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "you", "your", "this", "they", "their", "have",
    "had", "but", "not", "or", "we", "his", "her", "she", "who", "what",
    "when", "where", "why", "how", "about", "after", "all", "also", "can",
    "more", "one", "out", "up", "there", "which", "would", "said", "been",
}

LABEL_DISPLAY = {
    "FAKE": "SOXTA",
    "REAL": "HAQIQIY",
}


def clean_news_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in STOP_WORDS and len(word) > 2]
    return " ".join(words)


def top_words_for_label(news_data, label, limit=20):
    words = " ".join(news_data.loc[news_data["label"] == label, "clean_text"]).split()
    return Counter(words).most_common(limit)


def metric_card(title, value, note=""):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "color": "#5f6b7a"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "700"}),
            html.Div(note, style={"fontSize": "12px", "color": "#7a8594"}),
        ],
        style={
            "border": "1px solid #dfe3ea",
            "borderRadius": "8px",
            "padding": "16px",
            "backgroundColor": "#ffffff",
            "minWidth": "180px",
        },
    )


app = dash.Dash(__name__)
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_nse = pd.read_csv(DATA_DIR / "NSE-Tata-Global-Beverages-Limited.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=data.loc[:, ["Date", "Close"]].reset_index(drop=True).copy()
new_data["Close"] = new_data["Close"].astype(float)

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model(BASE_DIR / "saved_model.keras")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:].copy()
valid.loc[:, 'Bashoratlar']=closing_price
valid.loc[:, "Error"] = valid["Close"] - valid["Bashoratlar"]
valid.loc[:, "Absolute Error"] = valid["Error"].abs()

mae = valid["Absolute Error"].mean()
rmse = np.sqrt((valid["Error"] ** 2).mean())
accuracy_like = max(0, 100 - (mae / valid["Close"].mean() * 100))



df= pd.read_csv(DATA_DIR / "stock_data.csv")
news_df = pd.read_csv(BASE_DIR / "news.csv")
news_df["full_text"] = news_df["title"].astype(str) + " " + news_df["text"].astype(str)
news_df["clean_text"] = news_df["full_text"].apply(clean_news_text)
news_df["word_count"] = news_df["full_text"].str.split().str.len()
news_df["title_word_count"] = news_df["title"].astype(str).str.split().str.len()

label_counts = news_df["label"].value_counts().reset_index()
label_counts.columns = ["label", "count"]
label_counts["display_label"] = label_counts["label"].map(LABEL_DISPLAY)

fake_words = top_words_for_label(news_df, "FAKE")
real_words = top_words_for_label(news_df, "REAL")

sample_options = [
    {
        "label": f"{LABEL_DISPLAY.get(row.label, row.label)}: {row.title[:75]}",
        "value": int(index),
    }
    for index, row in news_df.head(100).iterrows()
]

app.layout = html.Div([
   
    html.H1("Birjada bozori tahlili va soxta xabarni aniqlash", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='NSE-TATAGLOBAL birja maʼlumotlari',children=[
            html.Div([
                html.H2("Haqiqiy yopilish narxi",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Haqiqiy malumot",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Haqiqiy yopilish narxlari',
                            xaxis={'title':'Sana'},
                            yaxis={'title':'Yopilish narxi (INR)'}
                        )
                    }

                ),
                html.H2("LSTM modeli bashorati",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Taxminiy malumot",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Bashoratlar"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Bashorat qilingan yopilish narxlari',
                            xaxis={'title':'Sana'},
                            yaxis={'title':'Yopilish narxi (INR)'}
                        )
                    }

                ),
                html.H2("Haqiqiy va bashorat qilingan yopilish narxi",style={"textAlign": "center"}),
                html.Div(
                    [
                        metric_card("MAE", f"{mae:.2f}", "Oʻrtacha bashorat xatosi"),
                        metric_card("RMSE", f"{rmse:.2f}", "Katta xatolar kuchliroq hisoblanadi"),
                        metric_card("Baholash", f"{accuracy_like:.1f}%", "Oddiy aniqlik ko‘rinishi"),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "justifyContent": "center",
                        "flexWrap": "wrap",
                        "margin": "20px 0",
                    },
                ),
                dcc.Graph(
                    id="actual-vs-predicted",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train.index,
                                y=train["Close"],
                                mode="lines",
                                name="Oʻrgatish davri yopilish narxi",
                                line={"color": "#2f80ed"},
                            ),
                            go.Scatter(
                                x=valid.index,
                                y=valid["Close"],
                                mode="lines",
                                name="Haqiqiy yopilish narxi",
                                line={"color": "#1b7f45"},
                            ),
                            go.Scatter(
                                x=valid.index,
                                y=valid["Bashoratlar"],
                                mode="lines",
                                name="LSTM Bashoratlar - yopilish narxi",
                                line={"color": "#d35400", "dash": "dash"},
                            ),
                        ],
                        "layout": go.Layout(
                            title="Oʻrgatish maʼlumotlari, haqiqiy qiymatlar va LSTM bashoratlari",
                            xaxis={"title": "Sana"},
                            yaxis={"title": "Yopilish narxi (INR)"},
                            hovermode="x unified",
                        ),
                    },
                ),
                html.H2("Vaqt bo‘yicha bashorat xatosi",style={"textAlign": "center"}),
                dcc.Graph(
                    id="prediction-error",
                    figure={
                        "data":[
                            go.Bar(
                                x=valid.index,
                                y=valid["Error"],
                                name="Haqiqiy - Bashorat",
                                marker={"color": valid["Error"].apply(lambda x: "#1b7f45" if x >= 0 else "#c0392b")},
                            )
                        ],
                        "layout": go.Layout(
                            title="Model qayerda ortiqcha yoki kam bashorat qilgan",
                            xaxis={"title": "Sana"},
                            yaxis={"title": "Bashorat xatosi"},
                        ),
                    },
                )                
            ])                


        ]),
        dcc.Tab(label='Kompaniyalar aksiya maʼlumotlari', children=[
            html.Div([
                html.H1("Aksiyalarning eng yuqori va eng past narxlari",
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Bozor savdo hajmi", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ]),
        dcc.Tab(label='Fake News tahlili', children=[
            html.Div([
                html.H1("Fake News maʼlumotlar toʻplami tahlili", style={"textAlign": "center"}),
                html.Div(
                    [
                        metric_card("Jami maqolalar", f"{len(news_df):,}", "news.csv dagi qatorlar"),
                        metric_card("Soxta maqolalar", f"{int(label_counts.loc[label_counts['label'] == 'FAKE', 'count'].iloc[0]):,}", "Yorliq = SOXTA"),
                        metric_card("Haqiqiy maqolalar", f"{int(label_counts.loc[label_counts['label'] == 'REAL', 'count'].iloc[0]):,}", "Yorliq = HAQIQIY"),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "justifyContent": "center",
                        "flexWrap": "wrap",
                        "margin": "20px 0",
                    },
                ),
                html.H2("Soxta va haqiqiy maqolalar soni", style={"textAlign": "center"}),
                dcc.Graph(
                    id="label-balance",
                    figure={
                        "data": [
                            go.Bar(
                                x=label_counts["display_label"],
                                y=label_counts["count"],
                                marker={"color": ["#c0392b" if label == "FAKE" else "#1b7f45" for label in label_counts["label"]]},
                            )
                        ],
                        "layout": go.Layout(
                            title="Maʼlumotlar toʻplami balansi",
                            xaxis={"title": "Yangilik yorligʻi"},
                            yaxis={"title": "Maqolalar soni"},
                        ),
                    },
                ),
                html.H2("Maqola uzunligi taqsimoti", style={"textAlign": "center"}),
                dcc.Graph(
                    id="length-distribution",
                    figure={
                        "data": [
                            go.Histogram(
                                x=news_df.loc[news_df["label"] == "FAKE", "word_count"],
                                name="FAKE",
                                opacity=0.65,
                                marker={"color": "#c0392b"},
                            ),
                            go.Histogram(
                                x=news_df.loc[news_df["label"] == "REAL", "word_count"],
                                name="REAL",
                                opacity=0.65,
                                marker={"color": "#1b7f45"},
                            ),
                        ],
                        "layout": go.Layout(
                            title="Soxta va haqiqiy maqolalarning uzunligi",
                            xaxis={"title": "Har bir maqoladagi soʻzlar"},
                            yaxis={"title": "Maqolalar soni"},
                            barmode="overlay",
                        ),
                    },
                ),
                html.H2("Eng ko‘p uchraydigan so‘zlar", style={"textAlign": "center"}),
                dcc.Graph(
                    id="common-words",
                    figure={
                        "data": [
                            go.Bar(
                                x=[count for word, count in fake_words],
                                y=[word for word, count in fake_words],
                                name="SOXTA",
                                orientation="h",
                                marker={"color": "#c0392b"},
                            ),
                            go.Bar(
                                x=[count for word, count in real_words],
                                y=[word for word, count in real_words],
                                name="HAQIQIY",
                                orientation="h",
                                marker={"color": "#1b7f45"},
                            ),
                        ],
                        "layout": go.Layout(
                            title="Tozalashdan keyingi eng ko‘p so‘zlar",
                            xaxis={"title": "Takrorlanish soni"},
                            yaxis={"title": "Soʻz", "autorange": "reversed"},
                            barmode="group",
                            height=720,
                        ),
                    },
                ),
                html.H2("Matn qanday tozalanishini ko‘rish", style={"textAlign": "center"}),
                dcc.Dropdown(
                    id="news-sample-dropdown",
                    options=sample_options,
                    value=sample_options[0]["value"],
                    style={"display": "block", "margin": "0 auto 20px", "width": "80%"},
                ),
                html.Div(id="news-cleaning-preview"),
            ], className="container"),
        ])


    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'Eng yuqori {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Eng past {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"{', '.join(str(dropdown[i]) for i in selected_dropdown)} uchun vaqt bo‘yicha eng yuqori va eng past narxlar",
            xaxis={"title":"Sana",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Narx (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Savdo hajmi {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"{', '.join(str(dropdown[i]) for i in selected_dropdown_value)} uchun vaqt bo‘yicha bozor savdo hajmi",
            xaxis={"title":"Sana",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Savdolar hajmi"})}
    return figure


@app.callback(Output('news-cleaning-preview', 'children'),
              [Input('news-sample-dropdown', 'value')])
def update_news_cleaning_preview(selected_index):
    row = news_df.loc[selected_index]
    original_preview = row["full_text"][:900]
    cleaned_preview = row["clean_text"][:900]
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Asl maqola matni"),
                    html.P(original_preview),
                ],
                style={
                    "border": "1px solid #dfe3ea",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(
                [
                    html.H3("ML uchun tozalangan matn"),
                    html.P(cleaned_preview),
                ],
                style={
                    "border": "1px solid #dfe3ea",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(
                [
                    metric_card("Yorliq", LABEL_DISPLAY.get(row["label"], row["label"]), "Maʼlumotdagi haqiqiy sinf"),
                    metric_card("Asl soʻzlar", f"{row['word_count']:,}", "Tozalashdan oldin"),
                    metric_card("Tozalangan soʻzlar", f"{len(row['clean_text'].split()):,}", "Tozalashdan keyin"),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "margin": "20px 0",
                },
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "16px",
            "padding": "0 20px 30px",
        },
    )


if __name__=='__main__':
    app.run(debug=True)
