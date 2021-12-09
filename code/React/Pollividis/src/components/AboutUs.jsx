import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React from "react";
import ImageCard from "./ImageCard";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));

const AboutUs = () => {
  const classes = useStyles();
  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={2} xs={2}>
                  <Leftbar />
              </Grid>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          About Us
                      </Typography>
                      <Typography style={{marginBottom:10}} variant="h6" color="textSecondary" component="p">
                          PolliVidis is developed by five Computer Science Students for their Senior Project.
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Ömer Ünlüsoy
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          İrem Tekin
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Elif Gamze Güliter
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Umut Ada Yürüten
                      </Typography>
                      <Typography style={{marginBottom:20}} variant="h5"  component="p">
                          Ece Ünal
                      </Typography>

                      <ImageCard img="https://images.pexels.com/photos/7319337/pexels-photo-7319337.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500"/>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
