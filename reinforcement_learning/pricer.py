import dateutil.tz
import pandas as pd
import os
import glob

class Pricer:
    """
    A class to handle electricity price data, including reading prices from a CSV file,
    retrieving the start datetime, and fetching the closest price for a given time.

    Attributes:
        tz (datetime.tz): Timezone for the electricity price data.
        prices_path (str): Path to the most recent CSV file containing electricity prices.
        electricity_prices_df (pd.DataFrame): DataFrame containing the electricity prices with datetime indexing.
    """
    
    def __init__(self, dataset_directory="../datasets"):
        """
        Initializes the Pricer class by setting up the timezone, finding the most recent 
        CSV file with electricity prices, and loading the prices into a DataFrame.

        Args:
            dataset_directory (str): The directory path where the dataset (CSV files) are stored.
        """
        self.tz = dateutil.tz.gettz("America/Los_Angeles")
        self.prices_path = self.find_latest_csv(dataset_directory)
        self.electricity_prices_df = self.set_prices_df()

    def find_latest_csv(self, directory):
        """
        Finds the most recent CSV file in the given directory based on modification time.

        Args:
            directory (str): Directory path to search for CSV files.

        Returns:
            str: The path to the most recent CSV file.

        Raises:
            FileNotFoundError: If no CSV files are found in the directory.
        """
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the directory.")
        
        # Find the most recent file based on modification time
        latest_file = max(csv_files, key=os.path.getmtime)
        return latest_file

    def set_prices_df(self):
        """
        Reads the most recent CSV file into a DataFrame and localizes the datetime column to the specified timezone.

        Returns:
            pd.DataFrame: A DataFrame with electricity prices and localized datetime column.
        """
        df = pd.read_csv(self.prices_path)
        df["date"] = pd.to_datetime(
            df["date"], format="%m/%d/%Y %I:%M:%S %p"
        ).dt.tz_localize(self.tz)
        return df

    def get_start_datetime(self):
        """
        Retrieves the earliest datetime in the electricity price DataFrame.

        Returns:
            pd.Timestamp: The earliest timestamp in the electricity price DataFrame, localized to the specified timezone.
        """
        return self.electricity_prices_df["date"].min().tz_convert(self.tz)

    def get_price(self, current_time):
        """
        Finds the electricity price closest to the given current time.

        Args:
            current_time (pd.Timestamp): The current time to find the closest electricity price.

        Returns:
            float: The electricity price corresponding to the closest time in the DataFrame.
        """
        closest_time_row = self.electricity_prices_df.iloc[
            (self.electricity_prices_df["date"] - current_time).abs().argsort()[:1]
        ]
        return closest_time_row["price"].values[0]
