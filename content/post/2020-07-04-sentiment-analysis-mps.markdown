---
title: Gloomsters or not? A Sentiment Analysis of MPs on Twitter
author: Paul Sharkey
date: '2020-08-07'
slug: sentiment-analysis-mps
categories: ['NLP', 'Sentiment Analysis', 'R']
description: 'Ever wondered who is the most positive MP on Twitter? In this post, a data-driven approach is used to answer this question'
summary: 'Ever wondered who is the most positive MP on Twitter? In this post, a data-driven approach is used to answer this question'
math: true
markup: mmark

---



We're living in a time of polarised opinion, where culture wars are becoming more prevalent and even *cough* government policy. We've also been told on several occasions that recent crises can be overcome with an optimistic outlook. In parliament, acknowledgement of failure has become a thing of the past, with ministers eager to shift attention away from their handling of the coronavirus pandemic and appearing to rewrite history in doing so. How is this relevant to a data science blog, you may ask? In light of the government's almost ideological optimism, I was interested to know if this was reflected in the data. Tools from natural language processing give us the opportunity to quantify the sentiment behind the words that people use. Speeches from the House of Commons are available for analysis, but these are often dominated by ministers and prominent MPs who are more concerned with showboating and party-political grandstanding than sincere communication of policy.

Twitter, in contrast, is used by most MPs and tends to be a more honest arbiter of their opinions and thoughts. In this post, we analyse the sentiments of MPs on Twitter and identify the most positive MPs who use this medium, as well as assessing whether the unabashed, and arguably, misplaced optimism of the Conservative Party is reflected in the tweets of their MPs.

# Retrieving Twitter Data

R's `rtweet` package provides an easy way to pull tweets from the Twitter API, subject to limits on the number of requests. For example, `rtweet` only allows a maximum of 3,200 tweets to be grabbed from the timeline of each requested user. If you prefer the RStudio interface over the Twitter UI, this library also allows you to post your own tweets through the R Console. The official `rtweet` [documentation](https://docs.ropensci.org/rtweet/) tells you how to get set up with the Twitter API (you need your own Twitter account).

This exercise requires the knowledge of the Twitter handles of every MP - I'm very thankful to the folks at [MPs on Twitter](https://www.mpsontwitter.co.uk/) for providing a table of handles, of which I only had to update a few due to recent changes. Note that these handles were correct at the time of publication and are subject to change.

These are the libraries being used for this exercise.


```r
library(rtweet)
library(vader)
library(tidyverse)
library(tidytext)
library(data.table)
```



Next we import the table of Twitter handles, select only the columns we need, and remove the '@' symbol from the list of handles.


```r
mps <-  fread('https://raw.githubusercontent.com/sharkeypg/Random/master/twitter_mps.csv', header = TRUE)

handles <-
  mps %>% select(Handle, Party, Constituency) %>% 
  mutate(Handle_stripped = tolower(str_remove_all(Handle, "@")))
```

We can pass these through to `get_timelines()`, where we ask to pull a maximum of 1000 tweets per MP.


```r
tmls <- get_timelines(handles$Handle_stripped, n = 1000)
```

The output of `get_timelines()` contains a plethora of fields - we only need a small number of these. We're not interested in what MPs retweet, but rather the tweets they create themselves. We also specify a date filter to restrict the dataset to tweets sent after the date of 2019 General Election.

Tweets are messy data sources and need some extensive tidying before we can do anything further. We remove any URLS, mentions, line breaks, special characters or emojis, ampersands, and any unnecessary blank space. We join the dataset back onto our original table to attach the party and constituency information for each MP. Piping this all together gives us the following. 


```r
url_regex <- "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

tmls_new <- 
  tmls %>% 
  dplyr::filter(is_retweet == FALSE,
                created_at > '2019-12-12') %>%
  mutate(text = str_remove_all(text, url_regex),
         text = str_remove_all(text, "#\\S+"),
         text = str_remove_all(text, "@\\S+"),
         text = str_remove_all(text, "\\n"),
         text = str_remove_all(text, '[\U{1f100}-\U{9f9FF}]'),
         text = str_remove_all(text, '&amp;'),
         text = str_remove_all(text, pattern = ' - '),
         text = str_remove_all(text, pattern = '[\u{2000}-\u{3000}]'),
         text = str_remove_all(text, pattern = '[^\x01-\x7F]'),
         text = str_squish(text),
         screen_name = tolower(screen_name)) %>% 
  left_join(handles, by = c('screen_name' = 'Handle_stripped')) %>%
  select(name, Party, Constituency, created_at, text)

glimpse(tmls_new)
```

```
## Rows: 193,383
## Columns: 5
## $ name         <chr> "Aaron Bell MP", "Aaron Bell MP", "Aaron Bell MP", "Aaro…
## $ Party        <chr> "Conservative", "Conservative", "Conservative", "Conserv…
## $ Constituency <chr> "Newcastle-under-Lyme", "Newcastle-under-Lyme", "Newcast…
## $ created_at   <dttm> 2020-08-04 09:42:22, 2020-07-20 14:55:05, 2020-07-19 09…
## $ text         <chr> "Im delighted to see the Governments scheme fund Innovat…
```

So what do MPs like to tweet about? Without going into too much detail on this, we can look at the top words MPs like to use when tweeting. The `tidytext` library allows us to wrangle with text data so easily - we can break each tweet down into individual words and remove what are known as 'stop words', common words ('the', 'and', 'or', etc) used to tie sentences together but are generally devoid of any meaning.


```r
words <- 
  tmls_new %>% 
  unnest_tokens(output = word, input = text) %>% 
  anti_join(stop_words) %>%
  count(word) %>% 
  top_n(10) %>% 
  arrange(desc(n))

words
```

```
## # A tibble: 10 x 2
##    word           n
##    <chr>      <int>
##  1 support    16824
##  2 people     15981
##  3 government 13473
##  4 local       9175
##  5 time        9088
##  6 uk          8420
##  7 workers     6376
##  8 day         6070
##  9 nhs         6021
## 10 news        5918
```

These top words aren't surprising and are consistent with what you'd expect from a parliamentary representative. Our main objective however, is to associate some measure of sentiment to a Tweet, so how exactly do we do this?

# Methodology

Sentiment analysis is a branch of natural language processing that detects and assigns a measure of polarity within text. This can be an incredibly useful tool in a business context; making it easier to summarise customer feedback and survey responses. Sentiment analysis models generally focus on determining whether a piece of text can be described as either positive, negative or neutral. There are two main types of algorithms used for this task:

* Rule-based approaches that give sentiment scores based on a set of manually-crafted rools
* Machine learning methods that learn from data to predict polarity

Given the absence of training data, I'll be using a state-of-the-art rule-based approach known as VADER. VADER, or Valence Aware Dictionary for sEntiment Reasoning, is a model that creates a dictionary mapping words and other features to sentiment expressions. It is able to capture both the polarity of opinion (positive, negative, neutral) and its intensity (distinguishing 'good' versus 'excellent' for example). It also makes use of other aspects of sentence structure to capture intensity, for example punctuation and capitalisation. A compound score is given for a text that ranges between -1 (extremely negative) and 1 (extremely positive). To illustrate this, let's compare two variations of the same sentence:


```r
get_vader('I am so very happy.')
```

```
##                                                    compound       pos       neu 
##   0.00000   0.00000   0.00000   0.00000   3.27135   0.64500   0.51600   0.48400 
##       neg but_count 
##   0.00000   0.00000
```


```r
get_vader('I AM SO VERY HAPPY!!!')
```

```
##                                                    compound       pos       neu 
##   0.00000   0.00000   0.00000   0.00000   3.27135   0.73100   0.56300   0.43700 
##       neg but_count 
##   0.00000   0.00000
```

The fact that in the second case, the text has been changed to all caps followed by some exclamation marks has increased the intensity of the positive sentiment and this is reflected in the compound score. For details on other criteria VADER uses to compute the compound score, the [original paper](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) is the best place to look.

# Who is the most positive MP on Twitter?

In VADER, we have a relatively simple but powerful methodology to determine the sentiment of tweets, especially since the scoring rules were established from social media blogs. Applying VADER to our collection of tweets is a relatively simple task. To mitigate the possibility that VADER may not recognise all lexical features contained in the tweets, we use an error catcher to return `NA` for a tweet that VADER cannot process.


```r
vader_compound <- function(x) {
  y <- tryCatch(tail(get_vader(x),5)[1], error = function(e) NA)
  return(y)
}

#vader
vader_scores <-
  tmls_new %>% 
  mutate(vader_scores = as.numeric(sapply(text, vader_compound )))
```

The most positive tweet from this sample was by the Tory MP for Keighley, [Robbie Moore](https://twitter.com/_robbiemoore), who said: *"Huge congrats to businesses that won their categories at Special to Joe Smith of for top prize of Winner of Winners! Awards are great success story for Ilkley brilliant means of showcasing our great businesses"*

The most negative tweet was written by Tory MP [David Duguid](https://twitter.com/david_duguid), who said: *"The original abuse is bad enough but then followed up by an elected councillor?There is a lot of stress and frustration on all sides within the political and social system but this kind of behaviour is not OK.No excuse for stress and frustration becoming hate and abuse."*

So who is the most positive MP on Twitter? To measure this, we filter out MPs that have written fewer than 30 tweets in the period studied and we take their mean VADER score.


```r
vader_mps <-
  vader_scores %>%
  group_by(name, Party) %>%
  summarise(total_tweets = n(),
            mean_score  = mean(vader_scores, na.rm = TRUE), .groups = 'drop') %>%
  dplyr::filter(total_tweets > 30) %>%
  top_n(10, mean_score) %>%
  arrange(desc(mean_score))
vader_mps
```

```
## # A tibble: 10 x 4
##    name                        Party        total_tweets mean_score
##    <chr>                       <chr>               <int>      <dbl>
##  1 David Rutley MP             Conservative           80      0.732
##  2 Rehman Chishti              Conservative          305      0.715
##  3 Kwasi Kwarteng MP           Conservative           43      0.707
##  4 Graham Stuart MP            Conservative          132      0.651
##  5 Alan Mak MP 🇬🇧              Conservative          140      0.627
##  6 Nadhim Zahawi #StayAlert    Conservative          100      0.624
##  7 Alberto Costa MP #StayAlert Conservative          425      0.621
##  8 Robin Walker                Conservative          216      0.615
##  9 Amanda Solloway             Conservative          272      0.614
## 10 Neil Hudson                 Conservative          194      0.609
```

The Tory MP for Macclesfield, [David Rutley](https://twitter.com/DavidRutley), is the winner. In fact, the Conservatives make up the entirety of the top 10.

# How do the parties compare?

We can compare the distribution of VADER scores across the main parties.


```r
vader_temp <- vader_scores %>% dplyr::filter(Party %in% c('Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat'))

ggplot(vader_temp) + 
  geom_boxplot(aes(x=Party, y = vader_scores)) +
  ylab('VADER score') +
  ggtitle('VADER scores for MPs tweets across the main UK political parties')
```

<img src="/post/2020-07-04-sentiment-analysis-mps_files/figure-html/unnamed-chunk-11-1.png" width="960" />

MPs of all political persuasions are more likely to convey positive rather than negative sentiment. While the overall distribution of VADER scores is similar across the parties, the Conservatives' median score is about 0.2 higher than the Labour party, which is unsurprising. The government's job is to champion policies that they're in a position to implement; the opposition's job is to scrutinise these policies and (usually) put forward the case as to why they will be bad for the country. It's reasonable to expect these roles to be reflected in the language of their respective MPs on Twitter. 

We can see how the party scores have evolved since the last general election.


```r
vader_time <- 
  vader_temp %>% 
  mutate(Date = as.Date(created_at)) %>% 
  group_by(Party, Date) %>% 
  summarise(mean_score = mean(vader_scores, na.rm = TRUE))

ggplot(vader_time, aes(x = Date, y = mean_score, group = Party, colour = Party)) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  ylab('VADER score') +
  ggtitle('Average VADER score by day, by party', subtitle = 'Local average also shown')
```

<img src="/post/2020-07-04-sentiment-analysis-mps_files/figure-html/unnamed-chunk-12-1.png" width="960" />

The above plot shows that MPs across all the main political parties are less positive on Twitter now compared to the end of last year. A pandemic will do that. The Conservatives were consistent in producing the most positive content since the election, though this suffered a marked drop in positivity over the course of this period as the pandemic hit. There has been a slight uptick in recent weeks however, likely a result of their attempt to restore confidence in the economy. The Labour party have consistently been the most negative, with a slight increase around the time of the leadership election, but has fallen back significantly in recent weeks, again likely as a result of their response to the government's questionable handling of the pandemic. The impact of a leadership election is also plain to see from the sharp increase in positive sentiment from the Liberal Democrats in recent weeks.

# Takeaways

VADER is a simple concept but clearly has involved countless hours of research into making a wonderfully powerful tool. This analysis has shown that indeed the Conservative party are a more optimistic bunch that their opponents, but when you're in government, that's your job, isn't it?

