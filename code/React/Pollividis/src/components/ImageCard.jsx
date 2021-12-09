import {
  Button,
  Card,
  CardActionArea,
  CardActions,
  CardContent,
  CardMedia,
  makeStyles,
  Typography,
} from "@material-ui/core";

const useStyles = makeStyles((theme) => ({
  card: {
    marginBottom: theme.spacing(5),
  },
  media: {
    height: 400,
    [theme.breakpoints.down("sm")]: {
      height:300,
    },
  },
}));

const SampleImagePreviewCard = ({ img}) => {
  const classes = useStyles();
  return (
      <Card className={classes.card}>
        <CardActionArea>
          <CardMedia className={classes.media} image={img}  />
        </CardActionArea>
      </Card>
  );
};

export default SampleImagePreviewCard;
