"""
    Trains & predicts an ESRNN model for a single product

    Parameters
    ----------
    train: DataFrame
        DataFrame with `config.DATE_COLUMN`, `config.IMPUTED_QTY` and `config.MEDICINE_ITEM_ID` columns at minimum.
    df_network: DataFrame
        
    product: str
        Unique identifier of medicine item to be forecasted
    validation: bool, optional
        Determines whether this is a validation or test run (default is False)

    Returns
    -------
    DataFrame
        Forecasted values for the given product
    """



"""Predicts for a fitted LSTM Model

    Parameters
    ----------
    model: model object
        Fitted LSTM model.

    scaler: ndarray of shape (n_features,)
        Per feature relative scaling of the data.

    medicine_item: string
        Medicine_item (Medicine_item Aggregate) for which forecast is being generated.

    df_network: Pandas DataFrame
        
    validation: Boolean

    """
