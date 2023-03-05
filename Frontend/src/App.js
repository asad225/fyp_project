import React from "react";

import { Switch, Route } from "react-router-dom";
import {
  Footer,
  Blog,
  Possibility,
  Features,
  WhatGPT3,
  Header,
} from "./containers";
import { CTA, Brand, Navbar } from "./components";
import "./App.css";
import ChatUI from "./components/UI/ChatUI";
import Chat from "./components/chat/Chat";

const App = () => (
  <Switch>
    <Route path='/' exact>
      <div className="App">
        <div className="gradient__bg">
          <Navbar />
          <Header />
        </div>
        <Brand />
        <WhatGPT3 />
        <Features />
        <Possibility />
        <CTA />
        <Blog />
        <Footer />
      </div>
    </Route>
    <Route path= '/livechat' >
      <ChatUI/>
    </Route>
    <Route path='/chat' >
        <Navbar/>
        <Chat/>
    </Route>
  </Switch>
);

export default App;
