import streamlit as st
import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

seed(123)

original_df = pd.read_csv('df-original.csv')
transformed_df = pd.read_csv('listings-transformed.csv')
pd.set_option('display.max_columns', len(transformed_df.columns))
pd.set_option('display.max_rows', 100)

# Separating X and y
X = transformed_df.drop(['price'], axis=1)
y = transformed_df.price

# Scaling
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train = X_train.fillna(1)
X_test = X_test.fillna(1)

param_grid = {'kernel': 'linear', 'C': 1.0}
model_svr = svm.SVR(**param_grid)
model_svr.fit(X_train, y_train)
#print(model_svr.score(X_test, y_test))

prediction = model_svr.predict(X_test)
prediction = np.exp(prediction)

st.sidebar.header('User Input Parameters')


def user_input_features(count):
    host_is_superhost = st.sidebar.radio("Is host superhost?",
                                         ('Yes', 'No'), key=count)
    host_identity_verified = st.sidebar.radio("Does host identity verified?",
                                              ('Yes', 'No'), key=count+1)
    host_has_profile_pic = st.sidebar.radio("Does host have a profile picture?",
                                            ('Yes', 'No'), key=count+2)
    host_total_listings_count = st.sidebar.slider('Host Total Listings', 1, 30, 2, key=count+3)
    bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 2, key=count+4)
    beds = st.sidebar.slider('Beds', 1, 10, 2, key=count+5)
    accommodates = st.sidebar.slider('Accommodates', 1, 10, 2, key=count+6)
    amenities_options = st.sidebar.multiselect("Select the amenities",
                                               ['Air conditioning', 'TV', 'Coffee machine',
                                                'Cooking basics', 'Dishwasher & Dryer & Washer',
                                                'Family/kid friendly', 'Parking', 'Nature and Views', 'Outdoor space'
                                                'Private entrance', 'Wifi', 'Long term stays', 'BBQ', 'Balcony',
                                                'High End Electronics', 'Bed Linen', 'Host greeting'], key=count+9)
    amenities = ''
    for amenity in amenities_options:
        amenities += amenity + ','
    amenities = amenities[:-1]
    review_scores_rating = st.sidebar.slider('Review Scores Rating', 0, 5, 4, key=count+7)
    neighbourhood = st.sidebar.radio("Which neighbourhood?",
                                     ['Oostelijk Havengebied - Indische Buurt', 'Centrum-Oost',
                                      'Centrum-West', 'Zuid', 'Oud-Oost', 'De Pijp - Rivierenbuurt',
                                      'Slotervaart', 'De Baarsjes - Oud-West', 'Bos en Lommer',
                                      'Geuzenveld - Slotermeer', 'IJburg - Zeeburgereiland',
                                      'Watergraafsmeer', 'Westerpark', 'Noord-Oost',
                                      'Buitenveldert - Zuidas', 'Oud-Noord', 'Noord-West',
                                      'De Aker - Nieuw Sloten', 'Osdorp', 'Bijlmer-Centrum',
                                      'Gaasperdam - Driemond', 'Bijlmer-Oost'], key=count+8)
    data = {
        'host_is_superhost': host_is_superhost,
        'host_identity_verified': host_identity_verified,
        'host_has_profile_pic': host_has_profile_pic,
        'host_total_listings_count': host_total_listings_count,
        'bedrooms': bedrooms,
        'beds': beds,
        'accommodates': accommodates,
        'amenities': amenities,
        'review_scores_rating': review_scores_rating,
        'neighbourhood': neighbourhood
    }
    features = pd.DataFrame(data, index=[0])
    return features


##########################

st.write("""
# AirBnb Price Prediction

This app predicts the AirBnb Prices!
""")

st.subheader('Original Data')
st.write(original_df)

st.subheader('Test Parameters')
st.write(transformed_df)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Predictions')
    st.write(prediction)

with col2:
    st.subheader('Real Values')
    st.write(np.exp(y_test))

#df_user_parameters = user_input_features(1)
#st.subheader('User Input Features')
#st.write(df_user_parameters)

columns = [X_test.columns]
X_test_original = pd.read_csv('xtest-original.csv')
data_user_parameter = X_test_original.loc[:1]
#index = data_user_parameter.id
#print(data_user_parameter)
df_user = user_input_features(50)

st.subheader('User Input Features')
st.write(df_user)

isSuperHost = 0
if df_user.host_is_superhost[0] == 'Yes':
    isSuperHost = 1
data_user_parameter.loc[0, 'host_is_superhost'] = isSuperHost


isIdentityVerified = 0
if df_user.host_identity_verified[0] == 'Yes':
    isIdentityVerified = 1
data_user_parameter.loc[0, 'host_identity_verified'] = isIdentityVerified

profilePic = 0
if df_user.host_has_profile_pic[0] == 'Yes':
    profilePic = 1
data_user_parameter.loc[0,'host_has_profile_pic'] = profilePic

data_user_parameter.loc[0,'host_total_listings_count'] = df_user.host_total_listings_count[0]

data_user_parameter.loc[0,'bedrooms'] = df_user.bedrooms[0]
data_user_parameter.loc[0,'bedrooms'] = df_user.beds[0]
data_user_parameter.loc[0,'bedrooms'] = df_user.accommodates[0]


data_user_parameter.loc[0,'air_conditioning'] = 0
data_user_parameter.loc[0,'high_end_electronics'] = 0
data_user_parameter.loc[0,'bbq'] = 0
data_user_parameter.loc[0,'balcony'] = 0
data_user_parameter.loc[0,'nature_and_views'] = 0
data_user_parameter.loc[0,'bed_linen'] = 0
data_user_parameter.loc[0,'tv'] = 0
data_user_parameter.loc[0,'coffee_machine'] = 0
data_user_parameter.loc[0,'cooking_basics'] = 0
data_user_parameter.loc[0,'white_goods'] = 0
data_user_parameter.loc[0,'child_friendly'] = 0
data_user_parameter.loc[0,'parking'] = 0
data_user_parameter.loc[0,'outdoor_space'] = 0
data_user_parameter.loc[0,'host_greeting'] = 0
data_user_parameter.loc[0,'internet'] = 0
data_user_parameter.loc[0,'long_term_stays'] = 0
data_user_parameter.loc[0,'private_entrance'] = 0


if 'Air Conditioning' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'air_conditioning'] = 1
if 'High End Electronics' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'high_end_electronics'] = 1
if 'BBQ' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'bbq'] = 1
if 'Balcony' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'balcony'] = 1
if 'Nature and Views' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'nature_and_views'] = 1
if 'Bed Linen' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'bed_linen'] = 1
if 'TV' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'tv'] = 1
if 'Coffee machine' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'coffee_machine'] = 1
if 'Cooking basics' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'cooking_basics'] = 1
if 'Dishwasher & Dryer & Washer' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'white_goods'] = 1
if 'Family/kid friendly' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'child_friendly'] = 1
if 'Parking' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'parking'] = 1
if 'Outdoor space' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'outdoor_space'] = 1
if 'Host greeting' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'host_greeting'] = 1
if 'Wifi' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'internet'] = 1
if 'Long term stays' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'long_term_stays'] = 1
if 'Private entrance' in str(df_user.amenities[0]):
    data_user_parameter.loc[0,'private_entrance'] = 1


data_user_parameter.loc[0,'review_scores_rating'] = df_user.review_scores_rating[0]


data_user_parameter.loc[0,'neighbourhood_cleansed_Bijlmer-Centrum'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Bijlmer-Oost'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Bos en Lommer'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Buitenveldert - Zuidas'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Centrum-Oost'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Centrum-West'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_De Aker - Nieuw Sloten'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_De Baarsjes - Oud-West'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_De Pijp - Rivierenbuurt'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Gaasperdam - Driemond'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Geuzenveld - Slotermeer'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_IJburg - Zeeburgereiland'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Noord-Oost'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Noord-West'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Oostelijk Havengebied - Indische Buurt'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Osdorp'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Oud-Noord'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Oud-Oost'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Slotervaart'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Watergraafsmeer'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Westerpark'] = 0
data_user_parameter.loc[0,'neighbourhood_cleansed_Zuid'] = 0


if df_user.neighbourhood[0] == 'Bijlmer-Centrum':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Bijlmer-Centrum'] = 1
elif df_user.neighbourhood[0] == 'Bijlmer-Oost':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Bijlmer-Oost'] = 1
elif df_user.neighbourhood[0] == 'Bos en Lommer':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Bos en Lommer'] = 1
elif df_user.neighbourhood[0] == 'Buitenveldert - Zuidas':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Buitenveldert - Zuidas'] = 1
elif df_user.neighbourhood[0] == 'Centrum-Oost':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Centrum-Oost'] = 1
elif df_user.neighbourhood[0] == 'Centrum-West':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Centrum-West'] = 1
elif df_user.neighbourhood[0] == 'De Aker - Nieuw Sloten':
    data_user_parameter.loc[0,'neighbourhood_cleansed_De Aker - Nieuw Sloten'] = 1
elif df_user.neighbourhood[0] == 'De Baarsjes - Oud-West':
    data_user_parameter.loc[0,'neighbourhood_cleansed_De Baarsjes - Oud-West'] = 1
elif df_user.neighbourhood[0] == 'De Pijp - Rivierenbuurt':
    data_user_parameter.loc[0,'neighbourhood_cleansed_De Pijp - Rivierenbuurt'] = 1
elif df_user.neighbourhood[0] == 'Gaasperdam - Driemond':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Gaasperdam - Driemond'] = 1
elif df_user.neighbourhood[0] == 'Geuzenveld - Slotermeer':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Geuzenveld - Slotermeer'] = 1
elif df_user.neighbourhood[0] == 'IJburg - Zeeburgereiland':
    data_user_parameter.loc[0,'neighbourhood_cleansed_IJburg - Zeeburgereiland'] = 1
elif df_user.neighbourhood[0] == 'Noord-Oost':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Noord-Oost'] = 1
elif df_user.neighbourhood[0] == 'Noord-West':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Noord-West'] = 1
elif df_user.neighbourhood[0] == 'Oostelijk Havengebied - Indische Buurt':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Oostelijk Havengebied - Indische Buurt'] = 1
elif df_user.neighbourhood[0] == 'Osdorp':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Osdorp'] = 1
elif df_user.neighbourhood[0] == 'Oud-Noord':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Oud-Noord'] = 1
elif df_user.neighbourhood[0] == 'Oud-Oost':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Oud-Oost'] = 1
elif df_user.neighbourhood[0] == 'Slotervaart':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Slotervaart'] = 1
elif df_user.neighbourhood[0] == 'Watergraafsmeer':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Watergraafsmeer'] = 1
elif df_user.neighbourhood[0] == 'Westerpark':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Westerpark'] = 1
elif df_user.neighbourhood[0] == 'Zuid':
    data_user_parameter.loc[0,'neighbourhood_cleansed_Zuid'] = 1

data_user_parameter = data_user_parameter.fillna(1.0)
X_test_user = pd.DataFrame(scaler.fit_transform(data_user_parameter), columns=list(data_user_parameter.columns))
prediction_user = model_svr.predict(X_test_user)
prediction_user = np.exp(prediction_user)


st.subheader('Price')
st.write(prediction_user[0])
