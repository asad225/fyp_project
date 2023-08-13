import React from 'react'
import './Chat.css'
import { useState } from 'react'
import { Link } from 'react-router-dom'

export default function Chat() {

    const [file,setFile] = useState()

    const fileHandler = (e) => {
        setFile(e.target.files[0])
    }
    const [messsage , setMessage] = useState('Your Model is training Please Wait')
    const [link , showLink] = useState(false)
    const [isTraining , setIsTraining] = useState(false)


    const formSubmitHandler = async (e) => {
        e.preventDefault()
        setIsTraining(true)
        const formData = new FormData();
        formData.append('file', file);
        try {
          const response = await fetch("http://127.0.0.1:8000/api/upload_csv", {
            method: "POST",
            body: formData,
          });
    
          const data = await response.json();
          console.log(data['message']);

          setMessage(data['message'])
          showLink(true)
        } catch (error) {
          console.error("Error uploading file:", error);
        }

        
    }

  return (
        <div className='chat-container'>
            <h1 className='gradient__text'>Upload your dataset file here</h1>
            <form action="" className='file-submit-form' onSubmit={formSubmitHandler}>
                <input type="file"  id='file' onChange={fileHandler}/>
               { <label htmlFor="file">Click to Upload</label>}
                {file && <p className='upload-status'>File Uploaded !</p>}
                {!link && <h1 className='gradient__text'>Click on button below for training</h1>}
                {!link && <button type='submit'>Train Dataset</button>}
            </form>
            {isTraining && <h1 className='gradient__text'>{messsage}</h1>}
            { link && <h1 className='gradient__text'><Link to='livechat'>Click here</Link></h1>}
        </div>
  )
}
