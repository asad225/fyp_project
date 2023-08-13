from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect , BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json
# from ask import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from preprocess import *
from train_ann import *






app = FastAPI()
is_training_complete = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# define endpoint


@app.get("/")
def Home():
    return "Welcome home"


def save_csv(file):
    try:
        file_path = os.path.join(os.getcwd(), file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return True, file_path
    except Exception as e:
        return False, str(e)







@app.post("/api/upload_csv")
async def upload_csv_file(background_tasks:  BackgroundTasks  , file: UploadFile = File(...) ):
    file_name = file.filename
    if file.filename.endswith(".csv"):
        success, file_path = save_csv(file)
        pre_process(file_name)
        train_model()
        is_training_complete = True
        background_tasks.add_task(start_websocket)  # Start WebSocket in the background

        if success:
            pre_process(file_name)
            return JSONResponse(content={"message": "Conrgatulations your model has been trained successfuly"})
        else:
            return JSONResponse(content={"message": "Error uploading and saving CSV file."}, status_code=500)
    else:
        return JSONResponse(content={"message": "Uploaded file is not a CSV file."}, status_code=400)



def start_websocket():
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: int):
        from chat import chatbot_response

        await manager.connect(websocket)
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        try:
            while True:
                data = await websocket.receive_text()
                #data which we got from frontend
                print(data)

                # respone = 'In Progress'
                # if is_training_complete:
                respone = chatbot_response(str(data))
                
                # respone = get_bot_response(str(data))
                respone = str(respone)
                # print(respone)
                # await manager.send_personal_message(f"You wrote: {data}", websocket)
                message = {"time":current_time,"clientId":client_id,"message":respone}
                # message = {"time":current_time,"clientId":client_id,"message":data}
                await manager.broadcast(json.dumps(message))
                
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            message = {"time":current_time,"clientId":client_id,"message":"Offline"}
            await manager.broadcast(json.dumps(message))