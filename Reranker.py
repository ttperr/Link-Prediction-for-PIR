import csv
import time
import numpy as np
from tqdm import tqdm
from graph import Graph


class Reranker(object):

    def __init__(self, dataset, validation=False):
        self.dataset = dataset
        self.validation = validation
        self.ranking_ratio = 1
        if dataset == "AOL":
            self.user_document = Graph(
                0, 5, self.validation, dataset=self.dataset, name="user-document"
            )
            self.UD_metrics = {}
            # ---- The following lines would load the query-document and query-user graphs. This is meant for further developments.
            # self.query_document = Graph(1, 5, self.validation, dataset=self.dataset, name="query-document")
            # self.query_user= Graph(1, 0, self.validation, dataset=self.dataset, name="query-user")

            self.METRICS = {  # ---- Lists the metrics computed to run the reranker. Uncomment more lines if you want to test more.
                # "user_document-shortest_distance": self.user_document.shortest_distance,
                # "user_document-weighted_shortest_distance": self.user_document.weighted_shortest_distance,
                # "user_document-common_neighbors": self.user_document.common_neighbors_metric,
                "user_document-adamic_adar": self.user_document.adamic_adar_metric,
                # "user_document-page_rank": self.user_document.rooted_page_rank,
                "user_document-prop_flow": self.user_document.prop_flow,
            }
        else:
            raise NotImplementedError("Unknown dataset")

    def updateGraphFromClicks(self, user_id, session, query_id, doc_id):
        """
        Updates the graphs from user clicks.
        Args:
            user_id: a string representing the user id. Possibly new.
            session: a string representing the session. It is relative to the user id.
            query_id: a string representing the query id. Possibly new.
            doc_id: a string representing the clicked document. Possibly new (wrt the graph, not the collection).
        """
        self.user_document.add_link(user_id, doc_id)
        # ---- The following lines are meant to be uncommented for implementations using query-document/query-user graphs.
        # self.query_document.add_link(query_id, doc_id)
        # self.query_user.add_link(query_id, user_id)

    def rerank(self, query_id, user, retrieved_docs, retrieved_scores, session=None):
        """
        Computes the reranking scores based on various arguments.

        Args:
            query_id: a string containing the query id.
            user: a string containing the user id. Can be new.
            retrieved_docs: a np.array of strings, representing the doc ids.
            retrieved_scores: a np.array of floats, containing the scores given by ElasticSearch. Index is the same as the docs. May be unsorted.
            session: a string containing the session id. Must support None, i.e. when the sesion is not provided.
        Returns:
            reranked_scores: a np.array of the same shape and type as retrieved_scores, with updated values.
        """
        if len(retrieved_docs) == 0:
            return None

        self.compute_UD_metrics(user, retrieved_docs)
        return self.mix_scores(  # The following lines lists the metrics used and their respective weights in the final ranking. Make sure to comment all metrics that does not appear in self.METRIC self.METRIC.
            (retrieved_scores, 1),
            # (self.UD_metrics["user_document-shortest_distance"], 1),
            # (self.UD_metrics["user_document-weighted_shortest_distance"], 1),
            # (self.UD_metrics["user_document-common_neighbors"], 1),
            (self.UD_metrics["user_document-adamic_adar"], 2),
            # (self.UD_metrics["user_document-page_rank"], 1),
            (self.UD_metrics["user_document-prop_flow"], 1),
        )

    def compute_UD_metrics(self, user, retrieved_docs, progress=False, metrics="all"):
        time_start = time.time()
        if metrics == "all":
            metrics = list(self.METRICS.keys())

        if progress:
            print("Computing user-document metrics")
        for metric in metrics:
            if progress:
                print(metric + "...", end="")
            self.UD_metrics[metric] = self.METRICS.get(metric)(user, retrieved_docs)
            if progress:
                print("done")
        if progress:
            print(f"Computations took {time.time()-time_start}s")

    def evaluation_metrics_scores(
        self,
        query_id,
        user,
        retrieved_docs,
        retrieved_scores,
        session=None,
        metrics="all",
    ):
        """
        Computes re-ranking scores based on all metrics chosen for evaluation.

        Args:
            query_id: a string containing the query id.
            user: a string containing the user id. Can be new.
            retrieved_docs: a np.array of strings, representing the doc ids.
            retrieved_scores: a np.array of floats, containing the scores given by ElasticSearch. Index is the same as the docs. May be unsorted.
            session: a string containing the session id. Must support None, i.e. when the session is not provided.
            metrics: a np.array containing the unique name of the metrics.
        Returns:
            re-ranked_scores: a np.array of the shape n_docs x metrics, containing the score for each doc. Must have the same order as the rows of metrics.
        """
        if len(retrieved_docs) == 0:
            return None
        if metrics == "all":
            metrics = list(self.METRICS.keys())
        self.compute_UD_metrics(user, retrieved_docs, metrics=metrics)
        return np.transpose(np.matrix([self.UD_metrics[metric] for metric in metrics]))

    def is_new_user(self, userID):
        """
        Must return true if the userId is new, false if it is known
        """
        return not self.user_document.nx_graph.has_node(userID)

    def PClick(self, queryID, retrieved_docs, retrieved_scores, userID):
        """Hello world metric, defined in https://dl.acm.org/doi/10.1145/1242572.1242651"""
        potentialMatches = 0
        matches = {}
        beta = 0.5
        res = []
        if self.dataset == "AOL":
            with open("datasets/AOL4PS/data.csv") as f:
                reader = csv.reader(f, delimiter="\t")
                firstRow = True
                pbar = tqdm(reader, desc="Parsing logs", unit="rows")
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    if row[1] == queryID and row[0] == userID:
                        potentialMatches += 1
                        matches[row[5]] = matches.get(row[5], 0) + 1
        else:
            raise NotImplementedError("Unknown dataset")
        for resultID, score in zip(retrieved_docs, retrieved_scores):
            newScore = (
                score * (1 - self.ranking_ratio)
                + (matches.get(resultID, 0) / (potentialMatches + beta))
                * self.ranking_ratio
            )
            res.append(newScore)
        return np.array(res)

    def mix_scores(self, *arg):
        """
        Mix re-ranking scores.

        Args:
            a list of (score, weight) where weight is an float and score is a np.array of shape n_docs x 1
        Returns:
            re-ranked_scores: a np.array of the shape n_docs x 1, containing the mixed score for each doc.
        """
        if len(arg) == 0:
            raise ValueError("No metric given.")
        shape = arg[0][0].shape
        res = np.zeros(shape=shape)
        for score, weight in arg:
            assert score.shape == shape
            norm = np.linalg.norm(score)
            normalized_score = (
                score / norm if norm != 0 else np.zeros(shape=score.shape)
            )
            res += weight * normalized_score
        return res

    def graph_features(self, query, user, documents, session=None):
        """
        Generate features for a given query to feed ML algorithm (unimplemented).
        """
        tmp = self.ranking_ratio
        self.ranking_ratio = 1
        k = len(documents)
        res = np.matrix(
            [
                # nodes degrees
                [self.user_document.degree(user)] * k,
                [self.user_document.degree(document) for document in documents],
                [self.session_document.degree(session)] * k,
                [self.session_document.degree(document) for document in documents],
                [self.query_document.degree(query)] * k,
                [self.query_document.degree(document) for document in documents],
                [self.query_user.degree(query)] * k,
                [self.query_user.degree(user)] * k,
                [self.query_session.degree(query)] * k,
                [self.query_session.degree(session)] * k,
                # shortest distance
                self.UD_shortest_distance,
                self.SD_shortest_distance,
                self.QD_shortest_distance,
                [self.QU_shortest_distance] * k,
                [self.QS_shortest_distance] * k,
                # weighted shortest distance (probably don't need both shortest distance metrics)
                self.UD_weighted_shortest_distance,
                self.SD_weighted_shortest_distance,
                self.QD_weighted_shortest_distance,
                [self.QU_weighted_shortest_distance] * k,
                [self.QS_weighted_shortest_distance] * k,
                # Common neighbors
                self.UD_common_neighbors,
                self.SD_common_neighbors,
                self.QD_common_neighbors,
                [self.QU_common_neighbors] * k,
                [self.QS_common_neighbors] * k,
                # Adamic Adar
                self.UD_adamic_adar,
                self.SD_adamic_adar,
                self.QD_adamic_adar,
                [self.QU_adamic_adar] * k,
                [self.QS_adamic_adar] * k,
                # page rank
                self.UD_page_rank,
                self.SD_page_rank,
                self.QD_page_rank,
                [self.QU_page_rank] * k,
                [self.QS_page_rank] * k,
                # prop flow
                self.UD_prop_flow,
                self.SD_prop_flow,
                self.QD_prop_flow,
                [self.QU_prop_flow] * k,
                [self.QS_prop_flow] * k,
            ]
        )
        self.ranking_ratio = tmp
        return res
