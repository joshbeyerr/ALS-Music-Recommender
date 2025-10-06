

from pathlib import Path

import scipy
import pandas as pd


def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    matrix = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    return matrix.tocsr()


class artistRetriever:
    def __init__(self):
        self.artistDF = None

    def getArtistName(self, artistID: int) -> str:
        if self.artistDF is None:
            raise ValueError("Artist data not loaded. Please load artist data first.")
        try:
            return self.artistDF.loc[artistID]["name"]
        except KeyError:
            return "Unknown Artist"

    def load_artist(self, artists_file: Path) -> None:
        artistDF = pd.read_csv(artists_file, sep="\t", index_col="id")
        self.artistDF = artistDF



if __name__ == "__main__":
    ret = artistRetriever()
    ret.load_artist(Path("lastfmdata/artists.dat"))

    print(ret.getArtistName(4))

    user_artists_matriix = load_user_artists(Path("lastfmdata/user_artists.dat"))

    print(user_artists_matriix)