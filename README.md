# Studio&Play

I'm working on a project title Studio&Play, which explores how artificial intelligence can serve as a tool to support us in the field of education. 

Studio&Play is a natural language processing (NLP) model that transforms study notes into lyrics of a song. It's meant to help the user make study notes easier to remember through mnemonics, thus boosting memory recall performance. 

The project is in its planning and developing stage:

## 1. "Translating" study notes into song lyrics
I first want to decide which model architecture to use. I'm considering a long-short term memory seq2seq model, or the transformer architecture. If it's the former, I'll employ the code that's offered by Coursera's course on sequence models as part of the deep learning specialization by Prof. Andrew Ng. If it's the latter I plan to work on the template code for the transformer implementation available at http://nlp.seas.harvard.edu/2018/04/03/attention.html. 

### 1.1 Training data
The training dataset is made up by me. It contains study notes for certain subject like Chemistry, and song lyrics belonging to a song adapted to some study notes that can be found at YouTube. They're separated by a ;

## 2. Melody composition out of the song lyrics
After the song lyrics are obtained, I want to pass it through a model discussed at https://deepai.org/publication/neural-melody-composition-from-lyrics or https://www.researchgate.net/publication/336022287_Komposer-Automated_Musical_Note_Generation_based_on_Lyrics_with_Recurrent_Neural_Networks to generate a melody out of them. 

The idea is that the model is multi-lingual, and can work for study notes that are in Chinese, Spanish and English. 

My project is open to be used for educational purposes, either by teachers of students, in order to make education more interactive and joyful. 

