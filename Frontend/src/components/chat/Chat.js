import React from 'react'
import './Chat.css'
import { useState } from 'react'
import { Link } from 'react-router-dom'

export default function Chat() {

    const [file,setFile] = useState()

    const fileHandler = (e) => {
        setFile(e.target.files[0])
    }


    const formSubmitHandler = (e) => {

    }

  return (
        <div className='chat-container'>
            <h1 className='gradient__text'>Upload your dataset file here</h1>
            <form action="" className='file-submit-form' onSubmit={formSubmitHandler}>
                <input type="file"  id='file' onChange={fileHandler}/>
                <label htmlFor="file">Click to Upload</label>
                {file && <p className='upload-status'>File Uploaded !</p>}
                <h1 className='gradient__text'>Click on button below for training</h1>
                <button type='submit'>Train Dataset</button>
                <p>Lorem ipsum dolor sit amet consectetur, adipisicing elit. Cumque repudiandae sequi, in ea architecto explicabo distinctio. Corporis neque eveniet nostrum, ducimus placeat sunt reiciendis, ipsa tempore ut porro officia quo!</p>
            </form>
            <h1 className='gradient__text'>Congratulations your training has completed.For live chat with bot <Link to='livechat'>Click here</Link></h1>
        </div>
  )
}
