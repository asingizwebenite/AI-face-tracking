"""
WebSocket Backend Service for Real-time Dashboard
Subscribes to MQTT movement messages and pushes updates to web clients via WebSocket
"""

import asyncio
import websockets
import json
import time
import paho.mqtt.client as mqtt
from typing import Set, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketBackend:
    def __init__(self, team_id: str, mqtt_broker: str = "157.173.101.159", 
                 mqtt_port: int = 1883, ws_port: int = 9002):
        self.team_id = team_id
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.ws_port = ws_port
        
        # MQTT topics
        self.movement_topic = f"vision/{team_id}/movement"
        self.heartbeat_topic = f"vision/{team_id}/heartbeat"
        
        # WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Latest data
        self.latest_movement = None
        self.latest_heartbeat = {}
        self.system_stats = {
            "last_update": time.time(),
            "total_messages": 0,
            "connected_clients": 0
        }
        
        # MQTT client
        self.mqtt_client = mqtt.Client(client_id=f"websocket_backend_{team_id}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
            # Subscribe to topics
            client.subscribe(self.movement_topic, qos=1)
            client.subscribe(self.heartbeat_topic, qos=0)
            logger.info(f"Subscribed to: {self.movement_topic}, {self.heartbeat_topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Add timestamp
            payload["mqtt_timestamp"] = time.time()
            
            if topic == self.movement_topic:
                self.latest_movement = payload
                self.system_stats["total_messages"] += 1
                self.system_stats["last_update"] = time.time()
                logger.info(f"Movement update: {payload.get('status', 'unknown')}")
                
            elif topic == self.heartbeat_topic:
                node = payload.get("node", "unknown")
                self.latest_heartbeat[node] = payload
                logger.info(f"Heartbeat from {node}")
            
            # Broadcast to all WebSocket clients
            asyncio.create_task(self._broadcast_to_clients({
                "type": "mqtt_message",
                "topic": topic,
                "payload": payload
            }))
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MQTT message: {e}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning("Disconnected from MQTT broker")
    
    async def _broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        self.system_stats["connected_clients"] = len(self.clients)
    
    async def handle_client(self, websocket, path):
        """Handle new WebSocket client connection"""
        logger.info(f"New client connected from {websocket.remote_address}")
        self.clients.add(websocket)
        self.system_stats["connected_clients"] = len(self.clients)
        
        try:
            # Send initial data to new client
            await websocket.send(json.dumps({
                "type": "initial_data",
                "latest_movement": self.latest_movement,
                "latest_heartbeat": self.latest_heartbeat,
                "system_stats": self.system_stats
            }))
            
            # Keep connection alive and handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)
            self.system_stats["connected_clients"] = len(self.clients)
    
    async def _handle_client_message(self, websocket, data: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = data.get("type")
        
        if message_type == "get_status":
            # Send current status
            await websocket.send(json.dumps({
                "type": "status_update",
                "latest_movement": self.latest_movement,
                "latest_heartbeat": self.latest_heartbeat,
                "system_stats": self.system_stats
            }))
        
        elif message_type == "ping":
            # Respond to ping
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }))
        
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }))
    
    def start_mqtt(self):
        """Start MQTT client in blocking mode"""
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client started")
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")
            raise
    
    async def start_websocket(self):
        """Start WebSocket server"""
        logger.info(f"Starting WebSocket server on port {self.ws_port}")
        
        # Start MQTT client
        self.start_mqtt()
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            "0.0.0.0",  # Listen on all interfaces
            self.ws_port
        )
        
        logger.info(f"WebSocket server started on ws://0.0.0.0:{self.ws_port}")
        logger.info(f"Team ID: {self.team_id}")
        logger.info(f"Monitoring MQTT topics: {self.movement_topic}, {self.heartbeat_topic}")
        
        await server.wait_closed()
    
    async def run(self):
        """Run the backend service"""
        try:
            await self.start_websocket()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Backend for Face-Locked Servo Dashboard")
    parser.add_argument("--team-id", required=True, help="Team ID for MQTT topics")
    parser.add_argument("--mqtt-broker", default="157.173.101.159", help="MQTT broker IP")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ws-port", type=int, default=9002, help="WebSocket server port")
    
    args = parser.parse_args()
    
    # Create and run backend
    backend = WebSocketBackend(
        team_id=args.team_id,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        ws_port=args.ws_port
    )
    
    logger.info("Starting WebSocket Backend Service")
    asyncio.run(backend.run())

if __name__ == "__main__":
    main()
