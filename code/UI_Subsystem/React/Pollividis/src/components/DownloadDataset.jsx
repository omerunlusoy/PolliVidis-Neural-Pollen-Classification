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

const DownloadDataset = () => {
  const classes = useStyles();
  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          Download Dataset
                      </Typography>
                      <a target="_blank" href="https://www.google.com/" title="example">
                          <Typography style={{marginBottom:10}} variant="h6" color="textSecondary" component="p">
                              Click here to download pollen dataset that Pollividis team photographed!
                          </Typography>
                      </a>

                      <ImageCard img="/irem_about_us.jpeg"/>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default DownloadDataset;
