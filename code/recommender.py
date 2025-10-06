from pathlib import Path
from typing import Tuple, List

import implicit
import scipy
from data import load_user_artists, artistRetriever



class Recommender:
    def __init__(self, artist_retriever: artistRetriever, implicit_model: implicit.recommender_base.RecommenderBase):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists: scipy.sparse.csr_matrix) -> None:
        # Train the implicit model
        self.implicit_model.fit(user_artists)

    # n is number of artist to recommend to user
    # returns a list of artists, each represented as a tuple of (artistID, score) where score represents the confidence of the recommendation
    def recommend(self, user_id: int, user_artists: scipy.sparse.csr_matrix, n: int = 10) -> Tuple[List[str], List[float]]:
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists[n], N=n
        )
        artists = [
            self.artist_retriever.getArtistName(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores
    

if __name__ == "__main__":

    # load user artists matrix
    user_artist = load_user_artists(Path("lastfmdata/user_artists.dat"))

    artistRetriever = artistRetriever()
    artistRetriever.load_artist(Path("lastfmdata/artists.dat"))

    implicit_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)

    recommender = Recommender(artistRetriever, implicit_model)
    recommender.fit(user_artist)
    artists, scores = recommender.recommend(2, user_artist, n=5)

    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")

    