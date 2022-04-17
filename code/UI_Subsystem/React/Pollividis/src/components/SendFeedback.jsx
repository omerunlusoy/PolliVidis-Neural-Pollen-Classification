import {
    Box,
    Card,
    CardActionArea,
    CardContent,
    CardMedia,
    Container,
    Dialog,
    Grid,
    makeStyles,
    Typography
} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import React, {useEffect, useState} from "react";
import ImageCard from "./ImageCard";
import TextField from '@mui/material/TextField';
import {Link} from "react-router-dom";
import Button from "@material-ui/core/Button";
import axios from "axios";

const useStyles = makeStyles((theme) => ({
  container: {
      justifyContent: 'center',
      alignItems: 'center',
    paddingTop: theme.spacing(10),
  },
}));

const AboutUs = () => {

    const [your_email, setEmail] = useState('');
    const [your_name, setName] = useState('');
    const [your_feedback, setFeedback] = useState('');
    const [open,setOpen] = useState(false);
    const [notEmpty,setNotEmpty] = useState(false);
    let academic_id;


    const academic = JSON.parse(sessionStorage.getItem('academic_id'));
    if(academic!= null)
    {
        academic_id = academic.academic_id;
    }
    else
    {
        academic_id = 0;
    }


    const handleClose = () => {
        setOpen(false);
    };


    const submitHandler= () => {

        const current = new Date();
        const date = `${current.getDate()}/${current.getMonth()+1}/${current.getFullYear()}`;

        let feedBackObject = new FormData(); // creates a new FormData object

        const myObject = {
            id: "",
            academic_id: academic_id,
            name: your_name,
            email: your_email,
            text: your_feedback,
            date: date,
            status: "",
        }



        feedBackObject.append("id",myObject.id);
        feedBackObject.append("academic_id",myObject.academic_id);
        feedBackObject.append("name", myObject.name); // add your file to form data
        feedBackObject.append("email",myObject.email);
        feedBackObject.append("text",myObject.text);
        feedBackObject.append("date",myObject.date);
        feedBackObject.append("status",myObject.status);


            axios
                .post('http://127.0.0.1:8000/api/feedback/', myObject)
                .then(response => {
                    console.log(response.data)

                })
                .catch(error => {
                    console.log(error)
                })

            console.log(myObject)

            setOpen(true);





    }

    const classes = useStyles();
  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          We value your feedback
                      </Typography>
                      <div style={{marginBottom:10}}>
                          <TextField
                              required
                              id="outlined-required"
                              label="Your name"
                              value={your_name}
                              onChange={e => setName(e.target.value)}
                          />
                      </div>
                      <div style={{marginBottom:10}}>
                          <TextField
                              required
                              id="outlined-required"
                              label="Your email"
                              value={your_email}
                              onChange={e => setEmail(e.target.value)}
                          />
                      </div>
                      <div style={{marginBottom:10}}>
                          <TextField
                              required
                              id="outlined-textarea"
                              label="Your Feedback"
                              placeholder="Your feedback"
                              multiline
                              rows={6}
                              value={your_feedback}
                              onChange={e => setFeedback(e.target.value)}
                          />
                      </div>
                      <div>
                          <Button type="submit" variant="contained" style={{backgroundColor:'#A6232A', color:'white'}}  onClick={submitHandler}>
                              Send Feedback
                          </Button>
                      </div>
                      <Dialog
                          open={open}
                          onClose={handleClose}
                      >
                          <Navbar/>
                          <Card >
                              <CardActionArea>
                                  <CardContent>
                                      <Typography align={"center"}  variant="h4" >
                                          Thank you for your feedback!
                                      </Typography>
                                  </CardContent>
                              </CardActionArea>
                          </Card>
                      </Dialog>

                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;


