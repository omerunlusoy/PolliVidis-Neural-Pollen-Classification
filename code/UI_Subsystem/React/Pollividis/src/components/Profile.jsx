import {
    Box,
    Button,
    Card,
    CardActionArea,
    CardContent,
    CardMedia,
    Container, Dialog,
    Grid,
    makeStyles, styled,
    Typography
} from "@material-ui/core";
import Navbar from "./Navbar";
import React, {useEffect, useState} from "react";
import ImageCard from "./ImageCard";
import ReactRoundedImage from "react-rounded-image";
import {getDownloadURL, ref, uploadBytesResumable} from "@firebase/storage";
import {storage} from "../firebase";
import axios from "axios";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
    right: {
        paddingRight: theme.spacing(10),
        paddingTop: theme.spacing(18),
    },
}));

const Profile = () => {
  const classes = useStyles();

    //TODO: render nothing if user is not logged in
    const id = JSON.parse(sessionStorage.getItem('academic_id'));
    console.log(id.academic_id)
    console.log(id.name)

    const [progress,setProgress] = useState(0)
    const [open, setOpen] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageUrl, setImageUrl] = useState("");
    const [photo, setPhoto] = useState("");

    let circleImage;


    const Input = styled('input')({
        display: 'none',
    });

    const handleClickOpen = () => {
        setOpen(true);
    };


    const uploadImage =(file,id)=>{
        if(!file) return;

        let fileU = '/profile/' + id
        const storageRef = ref(storage,fileU);
        const uploadTask = uploadBytesResumable(storageRef,file);

        uploadTask.on("state_changed",(snapshot)=>{
                const progress = Math.round((snapshot.bytesTransferred /snapshot.totalBytes) * 100);

                setProgress(progress)
            },(err) => {console.log(err)},
            () => {
                getDownloadURL(uploadTask.snapshot.ref).then(url => {console.log(url); })
            })

        setOpen(false);

    };


        let fileU = '/profile/' + (id.academic_id)
        const storageRef = ref(storage,fileU);
        console.log(fileU)
        getDownloadURL(storageRef).then((url)=>{
            setImageUrl(url)
        }).catch((error) => {
            // Handle any errors
            console.log(error)
        });


    const handleClose = () => {
        setOpen(false);
    };

    const handleDeleteImage = () => {
        setImageUrl(null);
        setPhoto(null)
        setOpen(false);
    };


    useEffect(() => {
        if (selectedImage) {
            setImageUrl(URL.createObjectURL(selectedImage));
            setPhoto(URL.createObjectURL(selectedImage));
        }
    }, [selectedImage]);



    if(imageUrl == "")
    {
        circleImage = "https://images.pexels.com/photos/7319337/pexels-photo-7319337.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500";
    }
    else
    {
        circleImage = imageUrl;
    }


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
                      <Typography style={{marginLeft:40, marginTop:50}}variant="h2" component="p">
                          Profile
                      </Typography>
                      <Typography style={{marginLeft:40, marginTop:15}} variant="h4"  component="p">
                          Name: {id.name} {id.surname}
                      </Typography>
                      <Typography style={{marginLeft:40, marginTop:15}} variant="h4"  component="p">
                          Job: {id.job_title}
                      </Typography>
                      <Typography style={{marginLeft:40, marginTop:15}} variant="h4"  component="p">
                          E-Mail: {id.email}
                      </Typography>
                      <Typography style={{marginLeft:40, marginTop:15}} variant="h4"  component="p">
                          Institution: {id.institution}
                      </Typography>
                      <Typography style={{marginLeft:40, marginTop:15}} variant="h4"  component="p">
                          Research Gate: {id.research_gate_link}
                      </Typography>
                      <Button variant="contained" style={{backgroundColor:'#A6232A', color:'white', marginTop:30, marginLeft:40, width:100, height:50}}  >
                          Edit
                      </Button>
                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right} style={{marginTop: 20}}>
                  <ReactRoundedImage
                      image={circleImage}
                      roundedColor="#321124"
                      imageWidth="250"
                      imageHeight="250"
                      roundedSize="10"
                      borderRadius="150"
                  />
                  <Button style={{marginTop:10}} variant="contained" style={{backgroundColor:'#A6232A', color:'white', marginTop:20, marginLeft:21}} onClick={handleClickOpen} size="medium" >
                      Change Profile Image
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
                                      {photo && selectedImage && (
                                          <Box mt={2} textAlign="center" style={{marginBottom:30}}>
                                              <img src={photo} alt={selectedImage.name} height="200px" />
                                              <div>
                                                  <label>
                                                      {selectedImage.name}
                                                  </label>
                                              </div>
                                          </Box>
                                      )}
                                  </div>
                                  <div align={"center"}>
                                      <Button  variant="contained"  style={{backgroundColor:'#A6232A', color:'white',marginRight:5}} size="medium" onClick={() => uploadImage(selectedImage,id.academic_id)} >
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
              </Grid>
          </Grid>
      </div>
  );
};

export default Profile;
