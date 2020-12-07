from river import stream

from . import base


class CreditCard(base.RemoteDataset):
    """Credit card frauds.

    The datasets contains transactions made by credit cards in September 2013 by european
    cardholders. This dataset presents transactions that occurred in two days, where we have 492
    frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class
    (frauds) account for 0.172% of all transactions.

    It contains only numerical input variables which are the result of a PCA transformation.
    Unfortunately, due to confidentiality issues, we cannot provide the original features and more
    background information about the data. Features V1, V2, ... V28 are the principal components
    obtained with PCA, the only features which have not been transformed with PCA are 'Time' and
    'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first
    transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be
    used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and
    it takes value 1 in case of fraud and 0 otherwise.

    References
    ----------
    [^1]: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
    [^2]: Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon
    [^3]: Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE
    [^4]: Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)
    [^5]: Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Ael; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier
    [^6]: Carcillo, Fabrizio; Le Borgne, Yann-Ael; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing
    [^7]: Bertrand Lebichot, Yann-Ael Le Borgne, Liyun He, Frederic Oble, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019
    [^8]: Fabrizio Carcillo, Yann-Ael Le Borgne, Olivier Caelen, Frederic Oble, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

    """

    def __init__(self):
        super().__init__(
            n_samples=150828752,
            n_features=30,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/creditcardfraud.zip",
            size=150828752,
            filename="creditcard.csv",
        )

    def _iter(self):

        converters = {f"V{i}": float for i in range(1, 29)}
        converters["Class"] = int
        converters["Time"] = float
        converters["Amount"] = float

        return stream.iter_csv(self.path, target="Class", converters=converters)
