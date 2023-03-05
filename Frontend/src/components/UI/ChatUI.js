import React, { useState,useEffect } from "react";
import "./styles.css";


const BOT_IMG = "https://image.flaticon.com/icons/svg/327/327779.svg";
const PERSON_IMG = "https://image.flaticon.com/icons/svg/145/145867.svg";
const BOT_NAME = "BOT";
const PERSON_NAME = "YOU";

function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

const ChatUI = () => {
  const [inputMessage, setInputMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      name: BOT_NAME,
      img: BOT_IMG,
      side: "left",
      text: "Hi, how can i help ya?",
    }
  ]);
  const [clientId, setClienId] = useState(
    Math.floor(new Date().getTime() / 1000)
  );
  const [websckt, setWebsckt] = useState();

  useEffect(() => {
    const url = "ws://localhost:8000/ws/" + clientId;
    const ws = new WebSocket(url);
    ws.onopen = (event) => {
      console.log("connected");
      // ws.send("Connected");
    };
    // recieve message every start page
    ws.onmessage = (e) => {
      const message = JSON.parse(e.data);
      setMessages([
        ...messages,
        {
          name: BOT_NAME,
          img: BOT_IMG,
          side: "left",
          text: message.message,
        },
      ]);
    };
    setWebsckt(ws);
    //clean up function when we close page
    return () => ws.close();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault()
    setMessages((prevState) => [
      ...prevState,
      {
        name: PERSON_NAME,
        img: PERSON_IMG,
        side: "right",
        text: inputMessage,
      },
    ]);

    let array = [
      ...messages,
      {
        name: PERSON_NAME,
        img: PERSON_IMG,
        side: "right",
        text: inputMessage,
      },
    ];
    websckt.send(inputMessage);
    setInputMessage("");
    botResponse(array);
  };

  const botResponse = (array) => {
    websckt.onmessage = (e) => {
      const message = JSON.parse(e.data);
      console.log(message);
      setMessages([
        ...array,
        {
          name: BOT_NAME,
          img: BOT_IMG,
          side: "left",
          text: message.message,
        },
      ]);
    };
  };
  // recieve message every send message

  useEffect(() => {
    const msgContainer = document.querySelector(".msger-chat");
    msgContainer.scrollTop += 500;
  }, [messages]);
  return (
    <div className="main">
      <div className="msger">
        <header className="msger-header">
          <div className="msger-header-title">
            <i className="fas fa-comment-alt" /> CHATBOT
          </div>
          <div className="msger-header-options">
            <span>
              <i className="fas fa-cog" />
            </span>
          </div>
        </header>
        <div className="msger-chat">
          {messages.map((message, index) => (
            <div key={index} className={`msg ${message.side}-msg`}>
              <div
                className="msg-img"
                style={{ backgroundImage: `url(${message.img})` }}
              />
              <div className="msg-bubble">
                <div className="msg-info">
                  <div className="msg-info-name">{message.name}</div>
                  <div className="msg-info-time">{formatDate(new Date())}</div>
                </div>
                <div className="msg-text">{message.text}</div>
              </div>
            </div>
          ))}
        </div>

        <form className="msger-inputarea" onSubmit={handleSubmit}>
          <input
            type="text"
            className="msger-input"
            placeholder="Enter your message..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
          />
          <button type="submit" className="msger-send-btn">
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatUI;
