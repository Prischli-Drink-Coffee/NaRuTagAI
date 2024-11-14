import { VStack } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import ContentSection from "../components/maincontent";
import TagSection from "../components/tagsectionmain";


const MainPage = () => {
  const { width } = useWindowDimensions();

  return (
    <VStack minH="100vh" align="center" justify="center" bg="#ffffff"
      padding={25}
      alignItems="flex-start"
      spacing="20px"
      flexGrow={1}
      width={ width }
    >

    <ContentSection />

    <TagSection />

    </VStack>
  );
};
export default MainPage;
