from sklearn.base import TransformerMixin, BaseEstimator

from band_model import model_sigma_E_0

'''
this module implements pipeline segments for curating datasets of intrinsic
thermoelectric transport coefficients.
'''


class SigmaE0(BaseEstimator, TransformerMixin):
    """
    Transformer for adding sigma_E_0 fields to a dataset.
    """

    def fit(self, conductivity, seebeck, temperature):
        """
        Assigns column headers for each of the properties.

        Args:
            conductivity (str) The column header for conductivity.
            seebeck (str) The column header for the Seebeck coefficient.
            temperature (str) The column header for temperature.
        """
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.temperature = temperature

    def transform(self, dataset, s=1):
        """
        Adds a column for sigma_E_0 (S/m) to the dataset.

        Notes:
            takes ~2 minues for a dataset with 37,000 entries.

        Args:
            dataset (DataFrame) A pandas DataFrame containing transport data.
            s (int) The transport exponent (mechanim, s=1 for band transport).
        """

        # collects sigma_E_0 values
        sigma_E_0 = []
        for index, row in dataset.iterrows():

            # collects meastured transport coefficients
            transport_data = [row[self.seebeck], row[self.conductivity],
                              row[self.temperature]]

            # computes sigma_E_0
            sigma_E_0.append(model_sigma_E_0(*transport_data, s=s))

        # returns dataset with additional column
        dataset['sigma_E_0'] = sigma_E_0
        return dataset


if __name__ == '__main__':
    from pandas import DataFrame

    # loads dataset
    dataset = DataFrame.from_csv(
        path=('../datasets/20190711/' +
              '20190711_preprocessing_interpolated_data.csv'),
        index_col=None
    )

    # transforms dataset
    pipe = SigmaE0()
    pipe.fit(conductivity='Electrical conductivity',
             seebeck='Seebeck coefficient',
             temperature='Temperature')
    dataset = pipe.transform(dataset).sort_values(
        by=['sampleid', 'Temperature']
    )

    # saves transformed dataset
    dataset.to_csv('../datasets/20190711/20190711_sigma_E_0.csv')
