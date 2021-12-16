//resim seçilmeden ya da resim silindikten sonra post yapılmasına ve sayfa değişmesine izin verme daha yapılmadı.

import {
    Box,
    Button,
    Card,
    CardActionArea,
    CardActions,
    CardContent,
    Container, Dialog,
    Grid, makeStyles, styled, TextField, Typography
} from "@material-ui/core";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React, {useEffect, useState} from "react";
import Navbar from "./Navbar";
import axios from 'axios';
import {Link, Navigate} from "react-router-dom";
import Leftbar from "./Leftbar";


const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));


const AnalyzeSample = () => {
  const classes = useStyles();
    const [open, setOpen] = React.useState(false);

    const handleClickOpen = () => {
        setOpen(true);
    };

    const Input = styled('input')({
        display: 'none',
    });

    const handleClose = () => {
        setOpen(false);
    };

    const [selectedImage, setSelectedImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [id, setId] = useState(null);
    const [goAnalysisPage, setGoAnalysisPage] = useState(false);

    const [date, setDate] = useState('');
    const [lat, setLat] = useState(null);
    const [lng, setLng] = useState(null);

    const [location, setLocation] = useState('');

    const handleDeleteImage = () => {
        setImageUrl(null);
        setOpen(false);
    };


    useEffect(() => {
        if (selectedImage) {
            setImageUrl(URL.createObjectURL(selectedImage));
        }
    }, [selectedImage,id]);

    let myObjectHelper = {
        id: "",
        date: "",
        lat: "",
        lng: "",
        image_url: ""
    }

    //analyze button handler
    const submitHandler= () => {

        const myObject = {
            id: "",
            date: date,
            lat: lat,
            lng: lng,
            image_url: imageUrl,
            analysis_text:"this is example analysis text"
        }


        axios
            .post('http://localhost:8000/analysis_posts', myObject)
            .then(response => {
                myObjectHelper = response;
                setId(myObjectHelper.data.id)
                setGoAnalysisPage(true)

            })
            .catch(error => {
                console.log(error)
            })

    }

    if(goAnalysisPage)
    {
            return <Navigate
                to={{
                    pathname: `/analysis/${id}`
                }}
            />

    }


  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Grid container>
                      <Grid item sm={8} >
                          <Container  className={classes.container}>
                              <Card  className={classes.card}>
                                  <CardActionArea>
                                      <CardContent>
                                              <Typography gutterBottom variant="h5" component="h2">
                                                  Analyze Sample
                                              </Typography>
                                          <CardActions>
                                                  <Button variant="contained" color="primary" size="medium" onClick={handleClickOpen}>
                                                      Upload Sample Image
                                                  </Button>
                                              <Dialog
                                                  fullScreen
                                                  open={open}
                                                  onClose={handleClose}
                                              >
                                                  <Navbar/>
                                                  <Card style={{marginTop:100}} className={classes.card}>
                                                      <CardActionArea>
                                                          <CardContent>
                                                              <Typography align={"center"} style={{marginBottom:30}} variant="h4" >
                                                                  Upload Image
                                                              </Typography>
                                                              <div align={"center"} style={{marginBottom:30}}>
                                                                  <label htmlFor="contained-button-file">
                                                                      <Input accept="image/*" id="contained-button-file" multiple type="file"  onChange={e => setSelectedImage(e.target.files[0])} />
                                                                      <Button variant="contained" component="span">
                                                                          Select Image
                                                                      </Button>
                                                                  </label>
                                                              </div>
                                                              <div>
                                                                  {imageUrl && selectedImage && (
                                                                      <Box mt={2} textAlign="center" style={{marginBottom:30}}>
                                                                          <img src={imageUrl} alt={selectedImage.name} height="200px" />
                                                                          <div>
                                                                              <label>
                                                                                  {selectedImage.name}
                                                                              </label>
                                                                          </div>
                                                                      </Box>
                                                                  )}
                                                              </div>
                                                              <div align={"center"}>
                                                                  <Button  variant="contained" style={{marginRight:5}} color="primary" size="medium" onClick={handleClose} >
                                                                      Use This Image
                                                                  </Button>
                                                                  <Button  variant="contained" color="secondary" size="medium" onClick={handleDeleteImage} >
                                                                      Delete This Image
                                                                  </Button>
                                                              </div>
                                                          </CardContent>
                                                      </CardActionArea>

                                                  </Card>
                                              </Dialog>
                                                  <Button variant="contained" color="primary"size="medium" >
                                                      Upload Image Collection
                                                  </Button>
                                          </CardActions>
                                          <CardActions>
                                                      <div>
                                                          <Button
                                                              variant="contained" color="primary" size="medium"
                                                              onClick={() => {
                                                                  navigator.geolocation.getCurrentPosition(
                                                                      (position) => {
                                                                          console.log("lat:",position.coords.latitude);
                                                                          console.log("lng:",position.coords.longitude);
                                                                          console.log(Date().toLocaleString());
                                                                          setDate(Date().toLocaleString());
                                                                          setLat(position.coords.latitude);
                                                                          setLng(position.coords.longitude);
                                                                      },
                                                                      () => null
                                                                  );

                                                              }}
                                                          > Get My Location </Button>
                                                      </div>
                                          </CardActions>
                                          <CardActions>
                                              <Button variant="contained" color="primary" size="medium" onClick={submitHandler}>Analyze</Button>
                                          </CardActions>
                                      </CardContent>

                                  </CardActionArea>

                              </Card>
                          </Container>
                      </Grid>
                      <Grid item sm={4} className={classes.right}>

                          <SampleImagePreviewCard img={imageUrl}/>
                      </Grid>
                  </Grid>
              </Grid>
              <Grid item sm={3} className={classes.right}>
              </Grid>
          </Grid>
      </div>
  );
};

export default AnalyzeSample;
