def remove_duplicates(data):
    '''
    The e-mail list has two files for each e-mail: from and to.
    For the enron scheme citation feature, it won`t matter if the citation was "from" a person or "to" a person.
    It will be consider either case, or the sum of both case, as the value for scheme citation.

    :param data: dataframe with e-mail list, with duplicates, and scheme citation for each
    :return: data: dataframe edited with e-mail list, without duplicates, and scheme citation
    '''

    for email in data.email_address.unique():
        schemes = data[data.email_address == email].schemes_citation.sum()
        data = data.append({'email_address' : email, 'schemes_citation' : schemes}, ignore_index=True)

    # drop the columns and just keep the last one created
    data.drop_duplicates('email_address', keep='last', inplace=True)

    # reset the index
    data.reset_index(drop=True, inplace=True)

    return data