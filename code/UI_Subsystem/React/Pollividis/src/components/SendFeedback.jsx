import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import React, {useState} from "react";
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

    const handleSubmit = e => {
        e.preventDefault();
        const feedback = {
            u_name: your_name,
            u_email: your_email,
            feedback: your_feedback
        };

        console.log(feedback)

    };

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
                          <Button type="submit" variant="contained" style={{backgroundColor:'#A6232A', color:'white'}}  onClick={handleSubmit}>
                              Send Feedback
                          </Button>
                      </div>

                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
