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
import React from "react";
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
                          Profile
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Name:
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Job:
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          E-Mail:
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          Institution:
                      </Typography>
                      <Typography style={{marginBottom:20}} variant="h5"  component="p">
                          Research Gate:
                      </Typography>
                      <Button style={{marginTop:10}} variant="contained" color="primary"size="medium" >
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
                  <Button style={{marginTop:10}} variant="contained" color="primary"size="medium" >
                      Change Profile Image
                  </Button>
              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
