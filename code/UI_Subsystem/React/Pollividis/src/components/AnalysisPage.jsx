import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React, {useEffect, useState} from "react";
import ImageCard from "./ImageCard";
import {useLocation, useParams} from "react-router-dom";
import {waitFor} from "@testing-library/react";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));

const Feed = () => {
  const classes = useStyles();

    const { id } = useParams();
    console.log("id:",id)


    const [analysis, setAnalysis] = useState([])
    const [isLoaded, setIsLoaded] = useState(false)

    useEffect(() => {
        fetch(`http://localhost:8000/api/analysis_get_id/${id}/`)
            .then((data) =>  data.json())
            .then((data) => setAnalysis(JSON.parse(data)))
            //.then((data) => console.log(data) )
    },[]);


  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          Analysis Report
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Location: {analysis.location_latitude}-{analysis.location_longitude}
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Date: {analysis.date}
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          {analysis.analysis_text}
                      </Typography>
                      <ImageCard img={analysis.sample_photo}/>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default Feed;
