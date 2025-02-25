import pickle

model = pickle.load(open('model.model', mode='rb'))
vectorizer = pickle.load(open('vectorizer.model', mode='rb'))

print('\n\n')
review = input('Enter the review: ')

prediction = model.predict(vectorizer.transform([review]))

if prediction == ['positive']:
    print('\n')
    print('It seems that the user liked the movie.')
    print('\n')

else:
    print('\n')
    print('It seems that the user did not like the movie.')
    print('\n')

print('')
