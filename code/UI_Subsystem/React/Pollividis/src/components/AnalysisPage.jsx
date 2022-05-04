import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";

import Navbar from "./Navbar";
import React, {useEffect, useState} from "react";
import ImageCard from "./ImageCard";
import {useLocation, useParams} from "react-router-dom";
import {waitFor} from "@testing-library/react";
import {storage} from "../firebase.js";
import {getDownloadURL, ref, uploadBytesResumable} from "@firebase/storage";

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
    const [photo,setPhoto] = useState(null)

    useEffect(() => {
        fetch(`http://localhost:8000/api/analysis_get_id/${id}/`)
            .then((data) =>  data.json())
            .then((data) => setAnalysis(JSON.parse(data)))
            //.then((data) => console.log(data) )
            .then(()=> getPhoto(id))
    },[]);


    const getPhoto = async(id) => {

        let fileU = '/files/' + (id) + '_final.jpg'
        const storageRef = ref(storage,fileU);
        getDownloadURL(storageRef).then((url)=>{
            console.log(url)
            setPhoto(url)
        }).catch((error) => {
            // Handle any errors
        });
    }


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
                          Analysis Text:
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          {analysis.analysis_text}
                      </Typography>
                      <img style={{
                          flex: 1,
                          width: '100%',
                          height: '100%',
                          resizeMode: 'contain',
                      }} src={photo}/>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default Feed;
