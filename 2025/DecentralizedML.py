import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from threading import Lock
import random

@dataclass
class ModelUpdate:
    """Represents a model update in the network"""
    node_id: str
    weights: np.ndarray
    timestamp: float
    prev_hash: str
    validation_score: float

@dataclass
class ConsensusBlock:
    """Represents a block of agreed-upon model updates"""
    updates: List[ModelUpdate]
    timestamp: float
    block_hash: str
    validator_signatures: List[str]

class DecentralizedNode:
    def __init__(self, node_id: str, initial_weights: np.ndarray):
        self.node_id = node_id
        self.current_weights = initial_weights
        self.peers: List[DecentralizedNode] = []
        self.update_history: List[ModelUpdate] = []
        self.consensus_chain: List[ConsensusBlock] = []
        self.pending_updates: List[ModelUpdate] = []
        self.lock = Lock()
        
    def compute_hash(self, data: Any) -> str:
        """Compute hash of data for consensus"""
        return hashlib.sha256(str(data).encode()).hexdigest()
    
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate a model update from another node"""
        # Check timestamp is reasonable
        if abs(time.time() - update.timestamp) > 3600:  # 1 hour max delay
            return False
            
        # Verify previous hash matches
        if self.update_history:
            expected_prev_hash = self.compute_hash(self.update_history[-1])
            if update.prev_hash != expected_prev_hash:
                return False
                
        # Validate weights format
        if update.weights.shape != self.current_weights.shape:
            return False
            
        # Check validation score is reasonable
        if not (0 <= update.validation_score <= 1):
            return False
            
        return True
    
    def train_local(self, data: np.ndarray, labels: np.ndarray) -> ModelUpdate:
        """Perform local training and create update"""
        with self.lock:
            # Simulate training with random weight updates
            weight_update = np.random.normal(0, 0.1, self.current_weights.shape)
            new_weights = self.current_weights + weight_update
            
            # Create model update
            update = ModelUpdate(
                node_id=self.node_id,
                weights=new_weights,
                timestamp=time.time(),
                prev_hash=self.compute_hash(self.update_history[-1]) if self.update_history else "",
                validation_score=self.compute_validation_score(new_weights, data, labels)
            )
            
            self.pending_updates.append(update)
            return update
            
    def compute_validation_score(self, weights: np.ndarray, data: np.ndarray, labels: np.ndarray) -> float:
        """Compute validation score for weights"""
        # Simulate validation score computation
        return random.uniform(0.7, 1.0)
    
    def propose_consensus_block(self) -> Optional[ConsensusBlock]:
        """Propose a new consensus block from pending updates"""
        with self.lock:
            if len(self.pending_updates) < 3:  # Need minimum number of updates
                return None
                
            # Sort updates by validation score
            valid_updates = [u for u in self.pending_updates if self.validate_update(u)]
            if not valid_updates:
                return None
                
            sorted_updates = sorted(valid_updates, key=lambda x: x.validation_score, reverse=True)
            
            # Take top performing updates
            selected_updates = sorted_updates[:5]
            
            # Create consensus block
            block = ConsensusBlock(
                updates=selected_updates,
                timestamp=time.time(),
                block_hash=self.compute_hash(selected_updates),
                validator_signatures=[self.node_id]
            )
            
            return block
            
    def validate_consensus_block(self, block: ConsensusBlock) -> bool:
        """Validate a proposed consensus block"""
        # Verify block timestamp
        if abs(time.time() - block.timestamp) > 3600:
            return False
            
        # Verify all updates are valid
        for update in block.updates:
            if not self.validate_update(update):
                return False
                
        # Verify block hash
        expected_hash = self.compute_hash(block.updates)
        if block.block_hash != expected_hash:
            return False
            
        # Verify validator signatures
        if not block.validator_signatures:
            return False
            
        return True
    
    def sign_consensus_block(self, block: ConsensusBlock):
        """Sign a valid consensus block"""
        if self.validate_consensus_block(block):
            if self.node_id not in block.validator_signatures:
                block.validator_signatures.append(self.node_id)
    
    def apply_consensus_block(self, block: ConsensusBlock):
        """Apply an agreed consensus block"""
        with self.lock:
            if not self.validate_consensus_block(block):
                return
                
            # Require majority of peers to have signed
            if len(block.validator_signatures) <= len(self.peers) / 2:
                return
                
            # Apply updates by averaging weights
            weight_updates = [update.weights for update in block.updates]
            self.current_weights = np.mean(weight_updates, axis=0)
            
            # Update chain
            self.consensus_chain.append(block)
            
            # Clear applied updates from pending
            pending_ids = {u.node_id for u in self.pending_updates}
            block_ids = {u.node_id for u in block.updates}
            self.pending_updates = [u for u in self.pending_updates 
                                  if u.node_id not in block_ids]
    
    def broadcast_update(self, update: ModelUpdate):
        """Broadcast update to peers"""
        for peer in self.peers:
            if peer.validate_update(update):
                peer.pending_updates.append(update)
    
    def broadcast_consensus_block(self, block: ConsensusBlock):
        """Broadcast consensus block to peers"""
        for peer in self.peers:
            peer.sign_consensus_block(block)
            if len(block.validator_signatures) > len(self.peers) / 2:
                peer.apply_consensus_block(block)

class DecentralizedML:
    """Coordinator for decentralized ML network"""
    
    def __init__(self, num_nodes: int, input_dim: int):
        # Initialize nodes with random weights
        self.nodes = []
        initial_weights = np.random.normal(0, 0.1, (input_dim,))
        
        for i in range(num_nodes):
            node = DecentralizedNode(f"node_{i}", initial_weights.copy())
            self.nodes.append(node)
            
        # Connect nodes in a peer network
        for node in self.nodes:
            node.peers = [p for p in self.nodes if p != node]
    
    def train_round(self, data: np.ndarray, labels: np.ndarray):
        """Perform one round of decentralized training"""
        # Each node performs local training
        for node in self.nodes:
            update = node.train_local(data, labels)
            node.broadcast_update(update)
            
        # Nodes propose and validate consensus
        for node in self.nodes:
            block = node.propose_consensus_block()
            if block:
                node.broadcast_consensus_block(block)

def main():
    # Example usage
    num_nodes = 5
    input_dim = 10
    
    # Create decentralized network
    network = DecentralizedML(num_nodes, input_dim)
    
    # Simulate training data
    data = np.random.normal(0, 1, (100, input_dim))
    labels = np.random.randint(0, 2, 100)
    
    # Run training rounds
    for round in range(10):
        print(f"Training round {round}")
        network.train_round(data, labels)
        
        # Print consensus chain lengths
        for i, node in enumerate(network.nodes):
            print(f"Node {i} consensus chain length: {len(node.consensus_chain)}")

if __name__ == "__main__":
    main()
