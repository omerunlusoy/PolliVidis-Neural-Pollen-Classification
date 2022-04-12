import {
    Button,
    Card,
    CardActionArea,
    CardContent,
    CardMedia,
    Container,
    Grid,
    makeStyles,
    Typography
} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React, {useEffect, useState} from "react";
import ImageCard from "./ImageCard";
import ReactRoundedImage from "react-rounded-image";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
    right: {
        paddingRight: theme.spacing(10),
        paddingTop: theme.spacing(18),
    },
}));

const AboutUs = () => {
  const classes = useStyles();

  //TODO: render nothing if user is not logged in
    const id = JSON.parse(sessionStorage.getItem('academic_id'));
    console.log(id.academic_id)
    console.log(id.name)
/*
    useEffect(() => {
        fetch(`http://localhost:8000/api/get_academic_by_id/${id}/`)
            .then((data) =>  data.json())
            .then((data) => setProfile(JSON.parse(data)))
        console.log(profile);
    },[]);*/

  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          Profile
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Name: {id.name} {id.surname}
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Job: {id.job_title}
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          E-Mail: {id.email}
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Institution: {id.institution}
                      </Typography>
                      <Typography style={{marginBottom:20}} variant="h5"  component="p">
                          Research Gate: {id.research_gate_link}
                      </Typography>
                      <Button style={{marginTop:10}} variant="contained" style={{backgroundColor:'#A6232A', color:'white'}}size="medium" >
                          Edit
                      </Button>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>
                  <ReactRoundedImage
                      image={"https://images.pexels.com/photos/7319337/pexels-photo-7319337.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500"}
                      roundedColor="#321124"
                      imageWidth="200"
                      imageHeight="200"
                      roundedSize="15"
                      borderRadius="100"
                  />
                  <Button style={{marginTop:10}} variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium" >
                      Change Profile Image
                  </Button>
              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
