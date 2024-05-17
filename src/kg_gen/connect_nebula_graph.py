"""
@Authors:
* Xianli Li (xli@assystem.com)
This script defines the Nebula_Graph class, providing a convenient interface for interacting with Nebula Graph.
It includes methods for initializing the Nebula Graph environment, creating a graph space, connecting to the graph,
executing queries, and saving graph data. The script leverages the Nebula2 Python client and other libraries
to facilitate seamless interactions with the Nebula Graph database.

Usage:
1. Create an instance of Nebula_Graph with the necessary connection details.
2. Call methods to perform various operations such as creating a graph space, connecting to the graph, executing queries,
   transforming results, and saving graph data.
"""
import os
import pandas as pd
from typing import Dict
import time
from nebula3.gclient.net import Connection
from nebula3.gclient.net.SessionPool import SessionPool
from nebula3.Config import SessionPoolConfig
from nebula3.common.ttypes import ErrorCode
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from nebula3.data.ResultSet import ResultSet

import networkx as nx
from pyvis.network import Network

# env variables
# os.environ['NEBULA_USER'] = "root"
# os.environ['NEBULA_PASSWORD'] = "nebula"
# os.environ["GRAPHD_HOST"] = "127.0.0.1"
# os.environ["GRAPHD_PORT"] = "9669"
# os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669"
# space_name = "Digital_Safety"
# space_name = "test"

class Nebula_Graph():
    """
    Nebula_Graph class facilitates interaction with Nebula Graph database.

    Attributes:
        nebula_user (str): Nebula Graph username.
        nebula_password (str): Nebula Graph password.
        graphd_host (str): Hostname of Nebula Graph server.
        graphd_port (str): Port number on which Nebula Graph server is running.
        space_name (str): Name of the Nebula Graph space. Default is "Digital_Safety".

    Methods:
        __init__(nebula_user, nebula_password, graphd_host, graphd_port, space_name):
            Constructor to initialize Nebula_Graph instance and create the graph space.

        creat_graph_space():
            Create the Nebula Graph space, session pool, and add schema.

        connect_nebula_graph() -> StorageContext:
            Connect to the Nebula Graph and return the storage context.

        result_to_dict(result: ResultSet) -> Dict:
            Transform the ResultSet to a dictionary.

        save_graph(as_html: bool = True) -> Optional[Dict]:
            Save the graph as an HTML file or return the graph data as a dictionary.

        close_session_pool():
            Close the Nebula Graph session pool.
    """
    def __init__(
        self, 
        nebula_user: str,
        nebula_password: str,
        graphd_host:str,
        graphd_port: str,
        space_name: str="Digital_Safety"
    ):
        self.nebula_user=nebula_user
        self.nebula_password=nebula_password
        self.graphd_host=graphd_host
        self.graphd_port=graphd_port
        self.space_name = space_name
        self.session_pool = None
        self.storage_context = None
        self.creat_graph_space()

    def creat_graph_space(self):
        """
        Create the Nebula Graph space, session pool, and add schema.
        """
        config = SessionPoolConfig()

        # prepare space
        conn = Connection()
        conn.open(self.graphd_host, self.graphd_port, 1000)
        auth_result = conn.authenticate(self.nebula_user, self.nebula_password)
        assert auth_result.get_session_id() != 0
        resp = conn.execute(
            auth_result._session_id,
            f"CREATE SPACE IF NOT EXISTS {self.space_name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);",
        )
        assert resp.error_code == ErrorCode.SUCCEEDED
        # insert data need to sleep after create schema
        time.sleep(10)

        self.session_pool = SessionPool(self.nebula_user, self.nebula_password, self.space_name, [(self.graphd_host, self.graphd_port)])
        assert self.session_pool.init(config)

        # add schema
        resp = self.session_pool.execute(
            'CREATE TAG IF NOT EXISTS entity(name string);'
            'CREATE EDGE IF NOT EXISTS relationship(relationship string);'
            'CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));'
        )
        assert resp.is_succeeded()

    def connect_nebula_graph(self):
        """
        Connect to the Nebula Graph and return the storage context.

        Returns:
            StorageContext: The storage context for Nebula Graph.
        """
        edge_types, rel_prop_names = ["relationship"], ["relationship"]
        tags = ["entity"]

        graph_store = NebulaGraphStore(
            space_name=self.space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )
        self.storage_context = StorageContext.from_defaults(graph_store=graph_store)
        return self.storage_context

    def result_to_dict(self, result: ResultSet) -> Dict:
        """
        Transform the ResultSet to a dictionary.

        Parameters:
            result (ResultSet): Nebula Graph ResultSet.

        Returns:
            Dict: Transformed dictionary containing graph data.
        """
        assert result.is_succeeded()
        result_dict = {}
        source = []
        target = []
        relation = []
        for row in result:
            relation_value = row.values()[0].get_value()
            source.append(relation_value.get_eVal().src.get_sVal().decode())
            target.append(relation_value.get_eVal().dst.get_sVal().decode())
            relation.append(relation_value.get_eVal().props[b'relationship'].get_sVal().decode())

        result_dict["source"] = source
        result_dict["target"] = target
        result_dict["relation"] = relation
        return result_dict

    def save_graph(self, as_html=True):
        """
        Save the graph as an HTML file or return the graph data as a dictionary.

        Parameters:
            as_html (bool): Save the graph as an HTML file. Default is True.

        Returns:
            Optional[Dict]: Graph data dictionary if not saving as HTML, otherwise None.
        """
        if self.session_pool != None:
            resp = self.session_pool.execute(
                'USE test;'
                'MATCH ()-[e]->() RETURN e'
            )
            result_dict = self.result_to_dict(resp)
            if as_html:
                # Create a directed graph
                graph = nx.DiGraph()

                # Add nodes and edges to the graph
                for src, tgt, rel in zip(result_dict['source'], result_dict['target'], result_dict['relation']):
                    graph.add_edge(src, tgt, relation=rel)

                net = Network(notebook=True, cdn_resources="in_line", directed=True)
                net.from_nx(graph)
                net.save_graph('./example.html')
                return None
            else:
                return result_dict
        else:
            print("Please initiate first the session pool.")
    
    def close_session_pool(self):
        """
        Close the Nebula Graph session pool.
        """
        self.session_pool.close()