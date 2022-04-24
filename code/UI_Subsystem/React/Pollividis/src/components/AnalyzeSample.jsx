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
import {storage} from "../firebase.js"
import {getDownloadURL, ref, uploadBytesResumable} from "@firebase/storage"
import Map from "./Map_v2";

//morphology sequence


const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));


const AnalyzeSample = () => {

    let academic_id;
    if (sessionStorage.getItem("academic_id") != null){
        const id_s = JSON.parse(sessionStorage.getItem('academic_id'));
        academic_id = id_s.academic_id;
    }
    else{
        academic_id = 1
    }

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

    const [progress,setProgress] = useState(0)
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    let id;
    const [goAnalysisPage, setGoAnalysisPage] = useState(false);
    const [myId,setMyId] = useState(null)

    const [date, setDate] = useState('');
    const [lat, setLat] = useState(null);
    const [lng, setLng] = useState(null);

    const [morp, setMorp] = useState("10");


    const uploadImage =(file,id)=>{
        if(!file) return;

        let fileU = '/files/' + id
        const storageRef = ref(storage,fileU);
        const uploadTask = uploadBytesResumable(storageRef,file);

        uploadTask.on("state_changed",(snapshot)=>{
            const progress = Math.round((snapshot.bytesTransferred /snapshot.totalBytes) * 100);
            setProgress(progress)
        },(err) => {console.log(err)},
            () => {
             getDownloadURL(uploadTask.snapshot.ref).then(url => {console.log(url);
                 axios.put('http://127.0.0.1:8000/api/analyze/', {url: url, id:id, morp:morp})
                     .then(response => {
                         // NAVIGATION
                         console.log(id)
                         setMyId(id)
                         setGoAnalysisPage(true)
                     })
             })
            })
    };

    const handleDeleteImage = () => {
        setImageUrl(null);
        setOpen(false);
    };


    useEffect(() => {
        if (selectedImage) {
            setImageUrl(URL.createObjectURL(selectedImage));
        }
    }, [selectedImage,id]);


    //analyze button handler
    const submitHandler= () => {

        let sampleObject = new FormData(); // creates a new FormData object

        const myObject = {
            sample_id: -1,
            academic_id: academic_id,
            sample_photo: selectedImage,
            date: date,
            location_latitude: lat,
            location_longitude: lng,
            analysis_text:"this is example analysis text",
            publication_status: false,
            anonymous_status:false,
            pollens: []
        }

        sampleObject.append("sample_id",myObject.sample_id);
        sampleObject.append("academic_id",myObject.academic_id);
        sampleObject.append("sample_photo", myObject.sample_photo); // add your file to form data
        sampleObject.append("date",myObject.date);
        sampleObject.append("location_latitude",myObject.location_latitude);
        sampleObject.append("location_longitude",myObject.location_longitude);
        sampleObject.append("analysis_text",myObject.analysis_text);
        sampleObject.append("publication_status",myObject.publication_status);
        sampleObject.append("anonymous_status",myObject.anonymous_status);
        sampleObject.append("pollens",myObject.pollens);



        axios
            .post('http://127.0.0.1:8000/api/analysis_posts/', sampleObject)
            .then(response => {
                console.log(response.data)
                id = response.data
                console.log(id)
                uploadImage(selectedImage,response.data)

            })
            .catch(error => {
                console.log(error)
            })


    }


    if(goAnalysisPage)
    {
            console.log("id",myId)
            return <Navigate
                to={{
                    pathname: `/analysis/${myId}`
                }}
            />

    }


  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10} >
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
                                                  <Button variant="contained" style={{backgroundColor:'#A6232A', color:'white'}}  size="medium" onClick={handleClickOpen}>
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
                                                                      <Input accept="image/*" id="contained-button-file" multiple type="file"  onChange={e => {setSelectedImage(e.target.files[0]);} }/>
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
                                                                  <Button  variant="contained"  style={{backgroundColor:'#A6232A', color:'white',marginRight:5}} size="medium" onClick={handleClose} >
                                                                      Use This Image
                                                                  </Button>
                                                                  <Button  variant="contained"  size="medium" onClick={handleDeleteImage} >
                                                                      Delete This Image
                                                                  </Button>
                                                              </div>
                                                          </CardContent>

                                                      </CardActionArea>

                                                  </Card>
                                              </Dialog>
                                                  <Button variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium" >
                                                      Upload Image Collection
                                                  </Button>
                                          </CardActions>
                                          <CardActions>
                                              <div>
                                                  <TextField id="outlined-basic" label="Morphology Sequence" variant="outlined" size={"small"} value={morp}
                                                             onChange={e => {setMorp(e.target.value); console.log(morp)}} />
                                              </div>
                                                      <div>
                                                          <Button
                                                              variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium"
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
                                              <Button variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium" onClick={submitHandler}>Analyze</Button>
                                          </CardActions>
                                      </CardContent>

                                  </CardActionArea>

                              </Card>
                              <Card>
                                  <CardActionArea>
                                      <CardContent>
                                          <Map/>
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
