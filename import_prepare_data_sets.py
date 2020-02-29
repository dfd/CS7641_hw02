import pandas as pd

ROOT = './candidate_datasets/'

def get_internet_ad_data():
    folder = ROOT + 'internet_ad_data/ad-dataset/'
    cols = ['height', 'width', 'aratio', 'local'] + ['X' + str(i) for i in range(1554)] + ['class']
    data = pd.read_csv(folder + 'ad.data',
                       header=None,
                       names=cols,
                       encoding='unicode_escape',
                       na_values=['?', ' ?', '  ?', '   ?', '    ?', '     ?'])
    #print(data.head(11))
    y = data['class']
    X = data.drop('class', axis=1)
    return X, y


def get_arsenic_data():
    folder = ROOT + 'arsenic/'
    data = pd.read_csv(folder + 'wells.dat.txt', sep=" ")
    #print(data.head())
    y = data['switch']
    X = data.drop('switch', axis=1)
    return X, y


def get_anxiety_data():
    folder = ROOT + 'anxiety/dataverse_files/'
    data = pd.read_excel(folder + 'Training Data.xlsx',
                         na_values='         .')
    #print(data.head())
    data = data.drop(['Subject', 'GAD', 'Sampling Weight',
                      'Onset Child tries unsuccessfully to leave daycare/school due to anxiety',
                      'Onset Has to be taken to daycare/school because of separation anxiety',
                      'Onset Child leaves daycare/school due to anxiety'
                      ], axis=1)
    y = data['SAD']
    X = data.drop('SAD', axis=1)
    return X, y


def get_credit_fraud_data():
    folder = ROOT + 'credit_fraud/'
    data = pd.read_csv(folder + 'creditcard.csv')
    data = data.drop('Time', axis=1)
    #print(data.head())
    y = data['Class']
    X = data.drop('Class', axis=1)
    return X, y


def get_telco_churn_data():
    folder = ROOT + 'teleco/'
    data = pd.read_csv(folder + 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
                       na_values=' '
                       )
    #print(data.head())
    data = data.drop('customerID', axis=1)
    data['TotalCharges'] = data['TotalCharges'].fillna(0)
    y = data['Churn']
    X = data.drop('Churn', axis=1)
    return X, y

def get_speed_dating_data():
    folder = ROOT + 'speed_dating/speed-dating-experiment/'
    data = pd.read_csv(folder + 'Speed Dating Data.csv',
                       encoding='unicode_escape')
    #print(data.head())
    data = data.drop(['iid', 'id'], axis=1)
    y = data['match']
    X = data.drop('match', axis=1)
    return X, y


def get_credit_financial_distress_data():
    folder = ROOT + 'credit_financial_distress/GiveMeSomeCredit/'
    data = pd.read_csv(folder + 'cs-training.csv',
                       index_col=0)
    #print(data.head(10))
    y = data['SeriousDlqin2yrs']
    X = data.drop('SeriousDlqin2yrs', axis=1)
    return X, y


def get_poisonous_mushrooms_data():
    folder = ROOT + 'poisonous_mushrooms/mushrooms/'
    cols = ['class',
            'cap-shape',
            'cap-surface',
            'cap-colour',
            'bruise',
            'odor',
            'attchmnt',
            'spacing',
            'size',
            'colour',
            'shape',
            'root',
            'sAbv',
            'sBelw',
            'cAbv',
            'cBelw',
            'vType',
            'vColour',
            'rNumber',
            'rType',
            'sColour',
            'pop',
            'habitat']

    data = pd.read_csv(folder + 'Dataset.data',
                       sep=' ',
                       names=cols,
                       header=None)
    #print(data)
    y = data['class']
    X = data.drop('class', axis=1)
    return X, y


def get_twonorm_data():
    folder = ROOT + 'twonorm/twonorm/'
    cols = ['I' + str(i) for i in range(1,21)] + ['class']
    data = pd.read_csv(folder + 'Dataset.data',
                       names=cols,
                       delim_whitespace=True,
                       header=None)
    #print(data)
    y = data['class']
    X = data.drop('class', axis=1)
    return X, y


def get_ringnorm_data():
    folder = ROOT + 'ringnorm/ringnorm/'
    cols = ['I' + str(i) for i in range(1,21)] + ['class']
    data = pd.read_csv(folder + 'Dataset.data',
                       names=cols,
                       delim_whitespace=True,
                       header=None)
    #print(data)
    y = data['class']
    X = data.drop('class', axis=1)
    return X, y


def get_splice_data():
    folder = ROOT + 'splice/splice/'
    cols = ['class'] + ['P_' + str(i) for i in range(30,0,-1)] + ['P' + str(i) for i in range(1, 31)]
    data = pd.read_csv(folder + 'Dataset.data',
                       header=None,
                       sep=' ',
                       names=cols
                       )
    #print(data.head())
    y = data['class']
    X = data.drop('class', axis=1)
    return X, y


# skip titanic for now, too much cleaning

def get_adult_data():
    folder = ROOT + 'adult/adult/'
    cols = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'educational-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'income'
            ]
    data = pd.read_csv(folder + 'Dataset.data',
                       header=None,
                       sep=' ',
                       names=cols
                       )
    #print(data.head())
    
    other_countries = [
        'Greece',
        'Nicaragua',
        'Peru',
        'Ecuador',
        'France',
        'Ireland',
        'Hong',
        'Thailand',
        'Cambodia',
        'Trinadad&Tobago',
        'Yugoslavia',
        'Outlying-US(Guam-USVI-etc)',
        'Laos',
        'Scotland',
        'Honduras',
        'Hungary',
        'Holand-Netherlands'
    ]

    data.loc[data['native-country'].isin(other_countries), 'native-country'] = 'Other'

    y = data['income']
    X = data.drop(['income', 'fnlwgt'], axis=1)
    return X, y


def get_travel_insurance_data():
    folder = ROOT + 'travel_insurance/'
    data = pd.read_csv(folder + 'travel insurance.csv')
    vc = data['Destination'].value_counts()
    c_list = list(vc.index[vc < 10])
    data.loc[data['Destination'].isin(c_list), 'Destination'] = 'Other'
    #print(data.head())
    y = data['Claim']
    X = data.drop('Claim', axis=1)
    return X, y


def get_facebook_survey_data():
    folder = ROOT + 'facebook_survey/'
    data = pd.read_csv(folder + 'FacebookNonUse-Qualtrics-2015.csv',
                       header=0,
                       skiprows=[1,2,3,4],
                       
                       )
    #print(data.head())
    data = data[data['Consent']=='Yes']
    y = data['HaveFB']
    X = data.drop(['HaveFB', 
                   'StartDate',
                   'EndDate',
                   'Progress',
                   'Duration (in seconds)',
                   'Finished',
                   'RecordedDate',
                   'ResponseId',
                   'Consent',
                   'Gender_4_TEXT'
                   ],
                   axis=1)
    #print(X.head())
    return X, y

def get_two_features_data():
    folder = ROOT + 'own/'
    data = pd.read_csv(folder + 'two_features.csv')
    y = data['y']
    X = data.drop('y', axis=1)
    return X, y

def get_two_features_plus_noise_data():
    folder = ROOT + 'own/'
    data = pd.read_csv(folder + 'two_features_plus_noise.csv')
    y = data['y']
    X = data.drop('y', axis=1)
    return X, y

def check_Xs():
    load_functions = [
        get_internet_ad_data,
        get_arsenic_data,
        get_anxiety_data,
        get_credit_fraud_data,
        get_telco_churn_data,
        #get_speed_dating_data,
        get_credit_financial_distress_data,
        get_poisonous_mushrooms_data,
        get_twonorm_data,
        get_ringnorm_data,
        get_splice_data,
        get_adult_data,
        get_travel_insurance_data,
        #get_facebook_survey_data
        get_two_features_data,
        get_two_features_plus_noise_data
    ]

    for i, func in enumerate(load_functions):
        print('i', i)
        X, y = func()
        cat = X.select_dtypes(include='object')
        if len(cat.columns) > 0:
            for col in cat.columns:
                if len(cat[col].unique()) > 15:
                    print('potential problem')
                    print(cat[col][0:5])
                    print(col)
                    print(len(cat[col].unique()))

        num = X.select_dtypes(exclude='object')            
        if len(num.columns) > 0:
            for col in num.columns:
                if num[col].isna().sum()/len(num) > .99:
                    print(col + ' col all missing')



if __name__ == "__main__":
    check_Xs()
    #X, y = get_travel_insurance_data()
    #vc = X['Destination'].value_counts()
    #print(vc.index[vc < 10])
    #print(X['native-country'].value_counts())
    #print(X['TotalCharges'])
    #for i in range(len(X['TotalCharges'])):
    #    if type(X['TotalCharges'][i]!=float):
    #        print(X['TotalCharges'][i])
