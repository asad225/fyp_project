import React from "react";
import "./cta.css";
import { useHistory } from "react-router-dom";

const CTA = () => {
  const history = useHistory()
  const redirectToChat = () => {
    history.push('./livechat')
  }
  return (
    <div className="gpt3__cta">
      <div className="gpt3__cta-content">
        <p>Request Early Access to Get Started</p>
        <h3>Register Today & start exploring the endless possibilities.</h3>
      </div>
      <div className="gpt3__cta-btn">
        <button type="button" onClick={redirectToChat}>Get Started</button>
      </div>
    </div>
  );
};
export default CTA;
