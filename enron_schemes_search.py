import glob
import os
import pickle
import pandas as pd

from get_schemes_citation import get_schemes
from parse_out_email_text_edited import parseOutText
from remove_repeted_emails import remove_duplicates

temp_counter = 0
from_email = []
num_schemes = []

# open all the .txt files on the emails_by_adress folder
for email_file in glob.glob('emails_by_address/*.txt'):

    # get the e-mail address that sent the message
    from_adrress = email_file.split('_')[3]
    from_email.append(from_adrress.split('.txt')[0])

    # initiate variable to store the number of scheme for each "from" e-mail
    schemes = 0

    # open the file and read the directory for the e-mails
    with open(email_file, 'r') as file:
        file.seek(0)
        path = file.read().split('\n')

        # for each directory, try to open the e-mail
        for directory in path:
            if len(directory) != 0:
                email_path = os.path.join('..', directory[len('enron_mail_20110402/'):-1]) #..\maildir/shapiro-r/all_documents/1618.
                try:
                    email = open(email_path, "r")
                except:
                    print(f'Error: {email_path}')

                # get the text using the function edited from the cloned git
                text = parseOutText(email)

                # get the number of schemes citation
                schemes = get_schemes(text, schemes)

    # store the number of schemes citation on the list associated to the e-mail
    num_schemes.append(schemes)

    # created just to get response that everything is running ok
    temp_counter += 1
    if (float(temp_counter / 100)) % 1 == 0:
        print(f'{temp_counter} emails analised')
        print('Still Scanning...')

# create a dataframe to store the result
df_schemes = pd.DataFrame({'email_address': from_email, 'schemes_citation': num_schemes})

# remove the duplicates e-mails in the dataframa
df_schemes = remove_duplicates(df_schemes)

# create a .txt file with the result to be imported
pickle.dump(df_schemes, open("enron_california_schemes.txt", "wb"))
