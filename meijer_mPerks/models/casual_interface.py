import econml

def estimate_uplift(test_df):
    model = econml.UpliftRandomForestClassifier(...)
    model.fit(y=test_df['redeemed'], T=test_df['treatment'], X=test_df[...])
    return model.effect(X=test_df[...])
