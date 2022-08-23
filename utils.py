import numpy as np
import pandas as pd
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import circlify as circ
import textwrap
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.colorway = [ '#60BE83', "#4c78a8","#f58518","#e45756","#59d8e0","#eeca3b","#b279a2","#ff9da6","#9d755d","#bab0ac"]
pio.templates[pio.templates.default].layout.font.family = "'Raleway', verdana, arial, sans-serif"

color_scheme = pio.templates[pio.templates.default].layout.colorway

stop_words = stopwords.words('english')
porter = PorterStemmer()
wnl = WordNetLemmatizer()
punct = string.punctuation + "“”’’"
transtable = str.maketrans('', '', punct)

kw_extractor = yake.KeywordExtractor(n=3, top=2)

def hex_to_rgba(hex, alpha):
    h = hex.lstrip('#')
    rgb = list(int(h[i:i+2], 16) for i in (0, 2, 4))
    rgb.append(alpha)
    return f"rgba({','.join([f'{b}' for b in rgb])})"

def clean_text(s):
    docs = []
    for i, comment in s.iteritems():
        if not pd.isna(comment):
            comment = comment.lower()
            words = word_tokenize(comment)
            newwords = []
            for w in words:
                w = w.translate(transtable)
                if w == 'nt': 
                    w = 'not'
                if len(w) > 0:
                    newwords.append(w)
            
            filtered_words = [word for word in newwords if word not in stop_words]
            stemmed = [wnl.lemmatize(word) for word in filtered_words]
            if len(stemmed) > 1:
                docs.append(stemmed)
                continue
        docs.append(None)

    return pd.Series(docs, index=s.index) 

def concat_series_text(s):
    
    concat = ""
    for comment in s:
        if comment:
            if type(comment) is str:
                concat += comment + " "
            elif comment is not None:
                concat += " ".join(comment) + " "
    return concat

def keyword_extractor(s):
    kws = []
    for i, comment in s.iteritems():
        if comment:
            keywords = kw_extractor.extract_keywords(" ".join(comment))
            if len(keywords) > 0:
                keywords = list(zip(*keywords))[0]
                kws.append(keywords)
                continue
        
        kws.append(None)
    return pd.Series(kws, index=s.index)

def process_text_col(df, col):
    cleaned_col = 'terms_' + col
    joined_col = 'joined_' + col
    keywords_col = 'keywords_' + col

    df[cleaned_col] = df[col].pipe(clean_text)

    cleaned = df[df[cleaned_col].notna()][cleaned_col].values
    raw = df[df[col].notna()][col].values

    phrase_model = Phrases(cleaned, min_count=2, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)

    def add_phrases(s):
        docs = []
        for i, comment in s.iteritems():
            if comment and len(comment) > 0:
                doc = phrase_model[comment]
                docs.append(doc)
                continue
            else:
                docs.append(None)
        return pd.Series(docs, index=s.index)

    df[cleaned_col] = df[cleaned_col].pipe(add_phrases)
    df[keywords_col] = df[cleaned_col].pipe(keyword_extractor)

    df[joined_col] = df[cleaned_col].apply(lambda x: " ".join(x) if x is not None else "")
    cleaned_corpus = df[df[joined_col] != ''][joined_col]
    raw_corpus = df[df[col] != ''][col]


    return cleaned_corpus, raw_corpus, df[cleaned_col], df[keywords_col]

def tfidf_cloud(corpus, preview=False, max_df=0.4, min_df=2):
    vectorizer = TfidfVectorizer(strip_accents='ascii', max_df=max_df, min_df=min_df, ngram_range = (1,1))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    feature_names

    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df.head()
    data = df.transpose()
    sums = np.mean(data, axis=1)

    # change the value to black
    def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl(0,100%, 1%)")

    # set the wordcloud background color to white
    # set max_words to 1000
    # set width and height to higher quality, 3000 x 2000
    width = 1200
    height = 1500
    if preview:
        width = 400
        height = 240
    wordcloud = WordCloud(background_color="white", width=width, height=height, max_words=500).generate_from_frequencies(sums)
    # set the word color to black
    wordcloud.recolor(color_func = black_color_func)


    # return cloud and tfidf matrix
    return wordcloud, sums

def save_cloud(wordcloud, filename, preview=False):
    # set the figsize
    if not preview:
        plt.figure(figsize=[12,15])
    else:
        plt.figure(figsize=[10,6])

    # plot the wordcloud
    # display(plt.imshow(wordcloud, interpolation="bilinear"))
    plt.imshow(wordcloud, interpolation='bilinear')
    # remove plot axes
    plt.axis("off")
    plt.savefig('wordclouds/' + filename)

def create_wordcloud_format(data):
    data = data[data.columns[:2]]
    title = data.columns[0]
    data.columns = ['word', 'weight']
    data = data[['weight', 'word']]
    data.to_csv(f'wordclouds/{title}.csv', index=False)

def word_cloud_pipeline(data, col, max_df=None, min_df=None, preview=False):
    cleaned_corpus, raw_corpus, _, _ = process_text_col(data, col)
    cloud, sums = tfidf_cloud(cleaned_corpus, preview=preview, max_df=max_df, min_df=min_df)
    # save_cloud(cloud, col+'.png', preview=preview)
    # cloud.cloud_to_file('worldclouds/' + col + '.png')
    if preview:
        plt.imshow(cloud, interpolation='bilinear')

    # prepare weights for wordcloud data export
    words = pd.DataFrame(sums).reset_index()
    words.columns = ['word', 'weight']
    words = words[['weight', 'word']]
    words.weight = preprocessing.minmax_scale(words.weight, feature_range=(1,len(words.index))).astype(int)
    words.to_csv('wordclouds/' + col + '.csv', index=False)

    return words

def explode_multiple_choice(data, col, delim='|'):
    res = data.copy()
    res[col] = res[col].apply(lambda x: [w.strip() for w in x.strip().split(delim) if w] if type(x) is str else x)

    return res.explode(col)

def score_hist(data, col, label, show_mean=True):
    pd = data.groupby(col).size().reset_index()
    pd.columns = [col, 'counts']
    pd['percentage'] = data.groupby(col).size().groupby(level=0).apply(lambda x: 100 * x / float(pd.counts.sum())).values
    pd['label'] = pd.apply(lambda x: '{0:2.0f}<br>({1:1.0f}%)'.format( x.counts, x.percentage), axis=1) 

    fig = px.bar(pd, x=col,y='counts', text='label')
    fig.update_layout(width=500, bargroupgap=0, bargap=0, xaxis_title=label, font_size=20
    )
    fig.update_traces(marker_line_color='black',
                    marker_line_width=1.5, opacity=1, textfont_size=15)

    if show_mean:
        mean = data[col].mean()
        fig.add_vline(x=mean, line_width=3, line_dash="dash", annotation_text=f"avg: {mean:1.2f}", annotation_position="top", line_color="black")

    return fig

def horizontal_bar_base(dtp, title, col):
    fig = px.bar(dtp, x='num', y=col, text='label', barmode='group', orientation='h')
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='auto', 
        marker_line_color='black',
        marker_line_width=0, opacity=1,
        textfont_size=18, 
                    # textfont_color='white'
    )
    fig.update_layout( yaxis={'categoryorder':'total ascending'}, yaxis_title='game', legend_title='Have Playdate yet?')
    fig.update_layout(
        title=title,
        xaxis_title='Number responses (% of responses)',
        autosize=True,
        height = 800,
        font_size=16
    )
    fig.update_yaxes(ticksuffix = "  ")
    return fig, dtp

def vertical_bar_base(dtp, title, col, col2=None):
    fig = px.bar(dtp, y='num', x=col, text='label', barmode='group')
    if col2:
        fig = px.bar(dtp, y='num', x=col, color=col2, text='label', barmode='group')
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='auto', 
        marker_line_color='black',
        marker_line_width=0, opacity=1,
        textfont_size=18, 
                    # textfont_color='white'
    )
    fig.update_layout( 
        xaxis={'categoryorder':'total descending'}, 
        yaxis_title='# Responses (% of responses)', legend_title='Have Playdate yet?')
    fig.update_layout(
        title=title,
        autosize=True,
        width = 1000,
        font_size=16
    )
    return fig, dtp

def horizontal_bar(data, title, col='fixed', other_threshold=3, total_responses=None, vertical=False,
    top_n=50, bot_n=None, col2=None):
    plot_data = data.copy()
    # col2 = 'have_playdate'
    groupby = [col] if not col2 else [col2, col]
    grouped = plot_data.groupby(col).size().reset_index()
    grouped = grouped.rename(columns={grouped.columns[-1]:'num'})
    group_in_other = grouped[grouped.num < other_threshold][col]
    plot_data['fixed'] = plot_data[col].apply(lambda x: 'other' if x in group_in_other.values else x)
    # plot_data['count'] = plot_data.timestamp.apply(lambda x: 'other' if x < 2 else x)
    plot_data = plot_data.groupby(groupby).size().reset_index()
    plot_data = plot_data.rename(columns={plot_data.columns[-1]:'num'})
    if total_responses: 
        plot_data['percentage'] = plot_data.num.apply(lambda x: 100 * (x / total_responses))
        plot_data['label'] = plot_data.apply(lambda x: f'{x.num:d} ({x.percentage:1.0f}%)', axis = 1)
    else:
        plot_data['label'] = plot_data.apply(lambda x: f'{x.num:d}', axis = 1)
    
    plot_data = plot_data.sort_values('num', ascending=False)
    if top_n and not bot_n:
        plot_data = plot_data.head(top_n)
    if bot_n and not top_n:
        plot_data = plot_data.tail(bot_n)

    if vertical:
        return vertical_bar_base(plot_data, title, col, col2)
    else:
        return horizontal_bar_base(plot_data, title, col)
        

def reformat_tag_data(data):
    # n_responses = data.comment.notna().sum()
    dtp = data.drop(columns='comment')
    n_responses = len(dtp.loc[~(dtp==0).all(axis=1)].index)
    dtp = dtp.sum().astype(int).to_frame().reset_index()

    dtp.columns = ['tag', 'num']

    dtp['percentage'] = dtp.num.apply(lambda x: 100 * (x / n_responses))
    dtp['label'] = dtp.apply(lambda x: f'{x.num:d} ({x.percentage:1.0f}%)', axis = 1)

    return dtp, n_responses

def horizontal_bar_tags(data, title):
    dtp, n_responses = reformat_tag_data(data)
    fig, dtp = horizontal_bar_base(dtp, title, col='tag')
    fig.update_traces(textposition='auto')
    return fig, dtp

def bubble_chart(data, title):
    dtp, n_responses = reformat_tag_data(data)
    dtp = dtp.sort_values(by='num', ascending=False, ignore_index=True)

    def format_label(row):
        tag = "<br>".join(textwrap.wrap(row.tag, width=12, break_long_words=False))
        return f'{tag}<br>{row.percentage:.1f}%'

    bubbles = circ.circlify(dtp.num.values.tolist())
    coords = list(zip(*[(b.x, b.y, b.r ) for b in bubbles]))
    dtp['x'] = coords[0][::-1]
    dtp['y'] = coords[1][::-1]
    dtp['r'] = coords[2][::-1]
    dtp['label'] = dtp.apply(format_label,axis=1)
    dtp['textsize'] = preprocessing.minmax_scale(dtp.num, feature_range=(15,40))

    fig = go.Figure()
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'layer':'below'}
    points = []
    for index, row in dtp.iterrows():
        x = row.x
        y = row.y
        r = row.r * 0.9
        opacity = 0.8 if index < len(color_scheme) else 1
        hexc = color_scheme[index % len(color_scheme)]
        color = hex_to_rgba(hexc, opacity)
        linecolor = hex_to_rgba(hexc, 1)
        shape = go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, fillcolor=color, line_color=linecolor, **kwargs)
        points.append(shape)


    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])
    fig.update_layout(
        width=1000, height=1000,
        shapes=points,
        xaxis=dict(showgrid=False, zeroline=False, showline=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, visible=False, showticklabels=False),
    )
    fig.add_trace(
        go.Scatter(
            x=dtp.x,
            y=dtp.y,
            text=dtp.label,
            textfont_size=dtp.textsize,
            mode="text",
        )
    )
    fig.update_traces(
        textfont_color='black'
    )

    return fig, dtp, n_responses
    
def pie(data, col, title=None, trace_order=None, horizontal=False, counted=False):
    if not title:
        title = " ".join(col.split("_"))
    plotd = data.copy()

    if trace_order:
        plotd[col] = pd.Categorical(
            plotd[col], categories=trace_order, ordered=True
        ) 
        plotd = plotd.sort_values(col, ascending=True)
    
    
    fig = px.pie(plotd, names=col, title=title)
    if counted:
        fig = px.pie(plotd, names=plotd.tag, values=plotd.num)
    fig.update_layout(width=380, height=380, legend_traceorder='normal')
    fig.update_traces(textfont_size=15, sort=trace_order is None)
    if horizontal:
        fig.update_layout(width=500, legend=dict(orientation="h", xanchor="center",x=0.5))
    return fig

def histogram(data, col, title, show_mean=True):
    pd = data.copy()
    fig = px.histogram(pd, x=col, nbins=20)
    fig.update_layout(width=500, bargroupgap=0, bargap=0, barmode='group', legend_title='Have purchased a game?', xaxis_title=title, 
        yaxis_title='responses',
        xaxis={
            'nticks':20,
            'tickmode':'auto'
        },
    )
    fig.update_traces(
        xbins_size=5,
        marker_line_color='black',
        marker_line_width=1.5, opacity=1, autobinx=False)
    
    if show_mean:
        mean = data[col].mean()
        fig.add_vline(x=mean, line_width=3, line_dash="dash", annotation_text=f"avg: {mean:1.2f}", annotation_position="top", line_color="black")


    return fig

def horizontal_marimekko(data, cols, top_labels, side_labels):
    cols = cols[::-1]
    x_data = np.zeros((len(cols), len(top_labels)))
    for i, col in enumerate(cols):
        for item, count in data[col].value_counts().items():
            j = top_labels.index(item)
            x_data[i, j] += count
    colors = color_scheme[0:len(top_labels)]
    x_data = np.round(x_data / 755 * 100,1)
    y_data = side_labels[::-1]

    fig = go.Figure()
    for i in range(0, len(x_data[0])): # for each top label
        first_color = True
        for xd, yd in zip(x_data, y_data): # for each y label
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1),
                ),
                showlegend=first_color,
                name=top_labels[i]
            ))
            first_color = False

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        # margin=dict(l=50, r=10, t=140, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.55,
            traceorder='normal'
        )
    )

    annotations = []
    perc_threshold = 4

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict( size=14,
                                        color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        if xd[0] > perc_threshold:
            # labeling the first percentage of each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=top_labels[0],#str(xd[0]) + '%',
                                font=dict(size=14,
                                        color='rgb(248, 248, 255)'),
                                showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                if xd[i] > perc_threshold:
                    annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i]/2), y=yd,
                                        text=top_labels[i],#str(xd[i]) + '%',
                                        font=dict(size=14,
                                                color='rgb(248, 248, 255)'),
                                        showarrow=False))
                space += xd[i]

    fig.update_layout(annotations=annotations, width=1500, font_size=16)
    return fig


def tag_similarity_matrix(data):
    sim_df = data.drop(columns='comment').astype(bool)
    sims = 1 - pairwise_distances(sim_df.T.values, metric = "jaccard")
    sims = pd.DataFrame(sims, index=sim_df.columns, columns=sim_df.columns)
    # corr = sdk_requests.drop(columns='comment').corr()
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = sims.columns,
            y = sims.index,
            z = sims.mask(mask),
            text=sims.values,
            colorscale=px.colors.sequential.deep,
            texttemplate='%{text:.2f}',
            zmin=0,
            zmax=1
        )
    )
    fig.update_layout(xaxis={'side': 'top'})
    return fig