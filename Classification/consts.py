import os


PATH_DATA = "data"
PATH_EMBED = os.path.join(PATH_DATA, "embeddings")


class EMBED_PROVIDER:
    DISTILBERT = "distilbert"
    TURKISH_BERT_TWEET = "turkishbertweet"
    OPENAI = 'openai-t3-large'


class MEAN_METHOD:
    MEAN = "mean"
    CENTROID = "w_centroid"


class TT:
    BIO = "ub_em"
    CAP = "uc_em"
    ALL = 'ue_em'


def get_filename(tt, provider, method=None, file_type="parquet", pca=0):

    if method:
        if pca > 0:
            return f"{PATH_EMBED}/{provider}/{tt}_{method}_pca[{pca}].{file_type}"
        else:
            return f"{PATH_EMBED}/{provider}/{tt}_{method}.{file_type}"
    else:
        return f"{PATH_EMBED}/{provider}/{tt}.{file_type}"

if __name__ == '__main__':
    a = get_filename(TT.CAP, EMBED_PROVIDER.DISTILBERT, MEAN_METHOD.MEAN)
    print(a)